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
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, CompositeSpec

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import ParallelEnv, TransformedEnv, ObservationNorm, Compose, \
    CatTensors, DoubleToFloat, EnvCreator, RewardSum

from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, ProbabilisticActor, SafeModule
from torchrl.modules.distributions import OneHotCategorical
from torch.distributions import Categorical

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator

from torchrl.objectives import SoftUpdate
from torchrl.trainers import Recorder



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


    # 1. Define environment --------------------------------------------------------------------------------------------

    # 1.1 Define env factory
    def env_factory(num_workers):
        """Creates an instance of the environment."""

        create_env_fn = lambda: GymEnv(env_name=args.task)

        # 1.2 Create env vector
        vec_env = ParallelEnv(create_env_fn=create_env_fn, num_workers=num_workers)

        # 1.3 Apply transformations to vec env - standard DeepMind Atari - Order of transforms is important!
        transformed_vec_env = TransformedEnv(vec_env)
        transformed_vec_env.append_transform(RewardSum())

        return transformed_vec_env

    # Sanity check
    test_env = env_factory(num_workers=3)
    test_input = test_env.reset()
    num_inputs = test_env.specs["observation_spec"]["observation"].shape[-1]
    num_actions = test_env.specs["action_spec"].space.n

    # Create Agent

    # Define Actor Network
    in_keys = ["observation"]
      
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": num_actions,
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    actor_module = SafeModule(
        #spec=CompositeSpec(action=test_env.specs["action_spec"]),
        module=actor_net,
        in_keys=in_keys,
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        spec=CompositeSpec(action=test_env.specs["action_spec"]),
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical, # OneHot
        distribution_kwargs={},
        default_interaction_mode="random",
        return_log_prob=False,
    ).to(device)

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": num_actions,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=in_keys,
        module=qvalue_net,
    ).to(device)
    
    # init nets
    with torch.no_grad():
        td = test_env.reset()
        td = td.to(device)
        actor(td)
        qvalue(td)

    del td
    test_env.close()
    test_env.eval()

    model = torch.nn.ModuleList([actor, qvalue])
    
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        action_spec=CompositeSpec(action=test_env.specs["action_spec"]),
        num_qvalue_nets=2,
        gamma=args.gamma,
        target_entropy_weight=args.target_entropy_weight,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, args.target_update_polyak)

    # Make Off-Policy Collector

    collector = SyncDataCollector(
        env_factory,
        create_env_kwargs={"num_workers": args.env_per_collector},
        policy=model[0],
        frames_per_batch=args.frames_per_batch,
        max_frames_per_traj=args.max_frames_per_traj,
        total_frames=args.total_frames,
        device=args.device,
        passing_device=args.device,
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
            episodes += torch.unique(tensordict["traj_ids"]).shape[0]
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
                    args.frames_per_batch * args.utd_ratio
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
                (i, args.frames_per_batch/torch.unique(tensordict["traj_ids"]).shape[0])
            )
            wandb.log(
                {   "unique traj": torch.unique(tensordict["traj_ids"]).shape[0],
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
            with set_exploration_mode("random"), torch.no_grad(): # TODO: WHY MEAN GIVES YOU NANS
                
                eval_rollout = test_env.rollout(max_steps=args.max_frames_per_traj, policy=actor,
                auto_cast_to_device=True,).clone()
                eval_reward = eval_rollout["reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                wandb.log({"test_reward": rewards_eval[-1][1]})
            if len(rewards_eval):
                pbar.set_description(
                    f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f})," + eval_str 
                )

        collector.shutdown()


def get_args():
    """Reads conf.yaml file in the same directory"""

    parser = argparse.ArgumentParser(description="RL")

    # Configuration file, keep first
    parser.add_argument("--conf", "-c", default="conf.yaml")
    parser.add_argument("--project", type=str, default="discrete_SAC_TorchRL")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="discrete_sac",
        help="Experiment name. Default: discrete_sac",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="CartPole-v1",
        help="MuJoCo training task. Default: CartPole-v1",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Number of frames to skip also called action repeat. Default: 1",
    )
    parser.add_argument(
        "--reward_scaling",
        type=float,
        default=1.0,
        help="Reward scaling factor. Default: 1.0",
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
        default=500,
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
    parser.add_argument("--target_entropy_weight", type=float, default=0.98, help="Target entropy weight. Default: 0.98")
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
    print("Using device: ", device)
    main()
