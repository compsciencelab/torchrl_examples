# Make all the necessary imports for training


import argparse
from typing import Optional

import numpy as np
import torch
import torch.cuda
import tqdm

import wandb
from sac_loss import SACLoss

from torch import nn, optim
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import (
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    ParallelEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, NormalParamWrapper, ProbabilisticActor, SafeModule
from torchrl.modules.distributions import TanhNormal

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator

from torchrl.objectives import SoftUpdate
from torchrl.trainers import Recorder


def make_env():
    """
    Create a base env
    """
    env_args = (args.task,)
    env_library = GymEnv
    
    env_kwargs = {
        "device": device,
        "frame_skip": args.frame_skip,
        "from_pixels": args.from_pixels,
        "pixels_only": args.from_pixels,
    }
    env = env_library(*env_args, **env_kwargs)
    return env


def make_transformed_env(
    env,
    stats=None,
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """

    env = TransformedEnv(env)

    # we append transforms one by one, although we might as well create the transformed environment using the `env = TransformedEnv(base_env, transforms)` syntax.
    env.append_transform(RewardScaling(loc=0.0, scale=args.reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []

    # We concatenate all states into a single "observation_vector"
    # even if there is a single tensor, it'll be renamed in "observation_vector".
    # This facilitates the downstream operations as we know the name of the output tensor.
    # In some environments (not half-cheetah), there may be more than one observation vector: in this case this code snippet will concatenate them all.
    selected_keys = list(env.observation_spec.keys())
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    #  we normalize the states
    if stats is None:
        _stats = {"loc": 0.0, "scale": 1.0}
    else:
        _stats = stats
    env.append_transform(
        ObservationNorm(**_stats, in_keys=[out_key], standard_normal=True)
    )

    double_to_float_list.append(out_key)
    env.append_transform(
        DoubleToFloat(
            in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
        )
    )

    return env


def parallel_env_constructor(
    stats,
    **env_kwargs,
):
    if args.env_per_collector == 1:
        env_creator = EnvCreator(
            lambda: make_transformed_env(make_env(), stats, **env_kwargs)
        )
        return env_creator

    parallel_env = ParallelEnv(
        num_workers=args.env_per_collector,
        create_env_fn=EnvCreator(lambda: make_env()),
        create_env_kwargs=None,
        pin_memory=False,
    )
    env = make_transformed_env(parallel_env, stats, **env_kwargs)
    return env


def get_stats_random_rollout(proof_environment, key: Optional[str] = None):
    print("computing state stats")
    n = 0
    td_stats = []
    while n < args.init_env_steps:
        _td_stats = proof_environment.rollout(max_steps=args.init_env_steps)
        n += _td_stats.numel()
        _td_stats_select = _td_stats.to_tensordict().select(key).cpu()
        if not len(list(_td_stats_select.keys())):
            raise RuntimeError(
                f"key {key} not found in tensordict with keys {list(_td_stats.keys())}"
            )
        td_stats.append(_td_stats_select)
        del _td_stats, _td_stats_select
    td_stats = torch.cat(td_stats, 0)

    m = td_stats.get(key).mean(dim=0)
    s = td_stats.get(key).std(dim=0)
    m[s == 0] = 0.0
    s[s == 0] = 1.0

    print(
        f"stats computed for {td_stats.numel()} steps. Got: \n"
        f"loc = {m}, \n"
        f"scale: {s}"
    )
    if not torch.isfinite(m).all():
        raise RuntimeError("non-finite values found in mean")
    if not torch.isfinite(s).all():
        raise RuntimeError("non-finite values found in sd")
    stats = {"loc": m, "scale": s}
    return stats


def get_env_stats():
    """
    Gets the stats of an environment
    """
    proof_env = make_transformed_env(make_env(), None)
    proof_env.set_seed(args.seed)
    stats = get_stats_random_rollout(
        proof_env,
        key="observation_vector",
    )
    # make sure proof_env is closed
    proof_env.close()
    return stats


def make_recorder(actor_model_explore, stats):
    base_env = make_env()
    recorder = make_transformed_env(base_env, stats)

    recorder_obj = Recorder(
        record_frames=1000,
        frame_skip=args.frame_skip,
        policy_exploration=actor_model_explore,
        recorder=recorder,
        exploration_mode="mean",
        record_interval=args.record_interval,
    )
    return recorder_obj


def make_replay_buffer(make_replay_buffer=3):
    if args.prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            args.buffer_size,
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=make_replay_buffer,
            storage=LazyMemmapStorage(
                args.buffer_size,
                scratch_dir=args.buffer_scratch_dir,
                device=device,
            ),
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            args.buffer_size,
            pin_memory=False,
            prefetch=make_replay_buffer,
            storage=LazyMemmapStorage(
                args.buffer_size,
                scratch_dir=args.buffer_scratch_dir,
                device=device,
            ),
        )
    return replay_buffer


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get stats for normalization
    stats = get_env_stats()

    # Environment setting:
    create_env_fn = parallel_env_constructor(
        stats=stats,
    )

    # Create Agent

    # Define Actor Network
    in_keys = ["observation_vector"]
    action_spec = create_env_fn.action_spec
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": False,
    }
    actor_net = NormalParamWrapper(
        actor_net,
        scale_mapping=f"biased_softplus_{1.0}",
        scale_lb=0.1,
    )
    in_keys_actor = in_keys
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        dist_in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # add forward pass for initialization with proof env
    proof_env = make_transformed_env(make_env(), stats)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    proof_env.close()

    actor_model_explore = model[0]

    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        gamma=args.gamma,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, args.target_update_polyak)

    # Make Off-Policy Collector

    collector = MultiaSyncDataCollector(
        create_env_fn=[create_env_fn],
        policy=actor_model_explore,
        total_frames=args.total_frames,
        max_frames_per_traj=args.env_per_collector*args.frames_per_batch,
        frames_per_batch=args.env_per_collector*args.frames_per_batch,
        init_random_frames=args.init_random_frames,
        reset_at_each_iter=False,
        postproc=None,
        split_trajs=True,
        devices=[device],  # device for execution
        passing_devices=[device],  # device where data will be stored and passed
        seed=None,
        pin_memory=False,
        update_at_each_batch=False,
        exploration_mode="random",
    )
    collector.set_seed(args.seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer()

    # Trajectory recorder
    recorder = make_recorder(actor_model_explore, stats)

    # Optimizers
    params = list(loss_module.parameters()) + list([loss_module.log_alpha])
    optimizer_actor = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()
    norm_factor_training = (
        sum(args.gamma**i for i in range(args.n_steps_forward))
        if args.n_steps_forward
        else 1
    )

    collected_frames = 0
    pbar = tqdm.tqdm(total=args.total_frames)
    r0 = None
    loss = None

    with wandb.init(project="SAC_TorchRL", name=args.exp_name, config=args):
        for i, tensordict in enumerate(collector):

            # update weights of the inference policy
            collector.update_policy_weights_()

            if r0 is None:
                r0 = tensordict["reward"].sum(-1).mean().item()
            pbar.update(tensordict.numel())

            # extend the replay buffer with the new data
            if "mask" in tensordict.keys():
                # if multi-step, a mask is present to help filter padded values
                current_frames = tensordict["mask"].sum()
                tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
            else:
                tensordict = tensordict.view(-1)
                current_frames = tensordict.numel()
            collected_frames += current_frames
            replay_buffer.extend(tensordict.cpu())

            # optimization steps
            if collected_frames >= args.init_random_frames:
                (
                    total_losses,
                    actor_losses,
                    q_losses,
                    alpha_losses,
                    alphas,
                    entropies,
                ) = ([], [], [], [], [], [])
                for _ in range(args.env_per_collector*args.optim_steps_per_batch):
                    # sample from replay buffer
                    sampled_tensordict = replay_buffer.sample(args.batch_size).clone()

                    loss_td = loss_module(sampled_tensordict)

                    actor_loss = loss_td["loss_actor"]
                    q_loss = loss_td["loss_qvalue"]
                    alpha_loss = loss_td["loss_alpha"]

                    loss = actor_loss + q_loss + alpha_loss
                    optimizer_actor.zero_grad()
                    loss.backward()
                    optimizer_actor.step()

                    # update qnet_target params
                    target_net_updater.step()

                    # update priority
                    if args.prb:
                        replay_buffer.update_priority(sampled_tensordict)

                    total_losses.append(loss.item())
                    actor_losses.append(actor_loss.item())
                    q_losses.append(q_loss.item())
                    alpha_losses.append(alpha_loss.item())
                    alphas.append(loss_td["alpha"].item())
                    entropies.append(loss_td["entropy"].item())

            rewards.append(
                (
                    i,
                    tensordict["reward"].sum().item()
                    / args.env_per_collector
                    / norm_factor_training
                    / args.frame_skip,
                )
            )
            wandb.log(
                {
                    "train_reward": rewards[-1][1],
                    "collected_frames": collected_frames,
                }
            )
            if loss is not None:
                wandb.log(
                    {
                        "total_loss": np.mean(total_losses),
                        "actor_loss": np.mean(actor_losses),
                        "q_loss": np.mean(q_losses),
                        "alpha_loss": np.mean(alpha_losses),
                        "alpha": np.mean(alphas),
                        "entropy": np.mean(entropies),
                    }
                )
            td_record = recorder(None)
            if td_record is not None:
                rewards_eval.append((i, td_record["total_r_evaluation"]))
                wandb.log({"test_reward": td_record["total_r_evaluation"]})
            if len(rewards_eval):
                pbar.set_description(
                    f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), test reward: {rewards_eval[-1][1]: 4.4f}"
                )

        collector.shutdown()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--exp_name", type=str, default="cheetah", help="Experiment name. Default: cheetah"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="HalfCheetah-v4",
        help="MuJoCo training task. Default: HalfCheetah-v4",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=2,
        help="Number of frames to skip also called action repeat. Default: 2",
    )
    parser.add_argument(
        "--from_pixels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pixel observations. Default: False",
    )
    parser.add_argument(
        "--reward_scaling",
        type=float,
        default=5.0,
        help="Reward scaling factor. Default: 5.0",
    )
    parser.add_argument(
        "--init_env_steps",
        type=int,
        default=1000,
        help="Initial environments steps used for observation stats computation. Default: 1000",
    )
    parser.add_argument(
        "--env_per_collector",
        type=int,
        default=5,
        help="Number of environments per collector device. Default: 5",
    )
    parser.add_argument(
        "--record_interval",
        type=int,
        default=10,
        help="Record interval for metrics logging based ob update iteration. Default: 10",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the experiment. Default: 42"
    )
    parser.add_argument(
        "--prb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Prioritized Experience Replay Buffer. Default: False",
    )
    parser.add_argument(
        "--n_steps_forward",
        type=int,
        default=None,
        help="Number of N-steps for the TD target calculation. Default: None (1 step)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Return discounting value. Default: 0.99",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100_000,
        help="Replay Buffer size. Default: 100.000",
    )
    parser.add_argument(
        "--buffer_scratch_dir",
        type=str,
        default="/tmp/",
        help="Buffer directory. Default: /tmp/",
    )
    parser.add_argument(
        "--max_frames_per_traj",
        type=int,
        default=-1,
        help="Maximum number of frames per rollout trajectory.",
    )
    parser.add_argument(
        "--frames_per_batch",
        type=int,
        default=1000,
        help="Number of frames per rollout batch. Default: 1000",
    )
    parser.add_argument(
        "--total_frames",
        type=int,
        default=1_000_000,
        help="Total number of frames. Default: 1_000_000",
    )
    parser.add_argument(
        "--init_random_frames",
        type=int,
        default=25000,
        help="Number of transitions taken by a random policy to prefill the Replay Buffer. Default: 25000",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size. Default: 256"
    )
    parser.add_argument(
        "--optim_steps_per_batch",
        type=int,
        default=1000,
        help="Number of updating steps after each data collection rollout. Default: 1000",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate. Default: 3e-4"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay. Default: 0.0"
    )
    parser.add_argument(
        "--target_update_polyak",
        type=float,
        default=0.995,
        help="Target network updating value for the soft-update. Default: 0.995",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Training device. Default: cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and args.device == "cuda:0"
        else torch.device("cpu")
    )
    main()
