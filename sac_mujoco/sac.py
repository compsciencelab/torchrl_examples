# Make all the necessary imports for training


import argparse
import yaml
from typing import Optional

import numpy as np
import torch
import torch.cuda
import tqdm

import wandb
from sac_loss import SACLoss

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    DoubleToFloat,
    ParallelEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, NormalParamWrapper, ProbabilisticActor, SafeModule
from torchrl.modules.distributions import TanhNormal

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator

from torchrl.objectives import SoftUpdate



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

    # Define env factory
    def env_factory(num_workers=1):
        """Creates an instance of the environment."""

        create_env_fn = lambda: GymEnv(env_name=args.task)

        vec_env = ParallelEnv(create_env_fn=create_env_fn, num_workers=num_workers)

        transformed_vec_env = TransformedEnv(vec_env)
        # transformed_vec_env.append_transform(RewardSum())
        transformed_vec_env.append_transform(
            DoubleToFloat(
                in_keys=["observation"], in_keys_inv=[]
            )
        )

        return transformed_vec_env

    test_env = env_factory(num_workers=1)

    # Create Agent

    # Define Actor Network
    in_keys = ["observation"]
    action_spec = test_env.action_spec
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
        in_keys=["loc", "scale"],
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

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = test_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    test_env.close()

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

    # Make Data Collector
    collector = SyncDataCollector(
        # we'll just run one ParallelEnvironment. Adding elements to the list would increase the number of envs run in parallel
        env_factory,
        create_env_kwargs={"num_workers": args.env_per_collector},
        policy=actor_model_explore,
        frames_per_batch=args.frames_per_batch,
        max_frames_per_traj=args.max_frames_per_traj,
        total_frames=args.total_frames,
    )
    collector.set_seed(args.seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer()

    # Optimizers
    params = list(loss_module.parameters()) + list([loss_module.log_alpha])
    optimizer_actor = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0
    episodes = 0
    pbar = tqdm.tqdm(total=args.total_frames)
    r0 = None
    loss = None

    with wandb.init(project=args.project, name=args.exp_name, config=args):
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
            episodes += args.env_per_collector
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
                for _ in range(
                    args.env_per_collector * args.frames_per_batch * args.utd_ratio
                ):
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
                (i, tensordict["reward"].sum().item() / args.env_per_collector)
            )
            wandb.log(
                {
                    "train_reward": rewards[-1][1],
                    "collected_frames": collected_frames,
                    "episodes": episodes,
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
            with set_exploration_mode("mean"), torch.no_grad():
                eval_rollout = test_env.rollout(args.max_frames_per_traj, actor_model_explore,
                auto_cast_to_device=True,)
                eval_reward = eval_rollout["reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                wandb.log({"test_reward": rewards_eval[-1][1]})
            if len(rewards_eval):
                pbar.set_description(
                    f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), " + eval_str 
                )

        collector.shutdown()


def get_args():
    """Reads conf.yaml file in the same directory"""

    parser = argparse.ArgumentParser(description="RL")

    # Configuration file, keep first
    parser.add_argument("--conf", "-c", default="conf.yaml")
    parser.add_argument("--project", type=str, default="SAC_TorchRL")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="cheetah",
        help="Experiment name. Default: cheetah",
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
        default=1,
        help="Record interval for metrics logging based ob update iteration. Default: 1",
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
        default=1000,
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
        "--utd_ratio",
        type=int,
        default=1,
        help="Update-to_Data ratio. For off policy algorithms we want to take at least one updating step each transition sampled. Default: 1",
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
    
    # Update args with conf.yaml
    if args.conf.endswith('yaml'):
        with open(args.conf) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        args.__dict__.update(config)

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
