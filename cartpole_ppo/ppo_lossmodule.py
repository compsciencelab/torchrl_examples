import math
import time
import yaml
import wandb
import torch
import argparse
import itertools

# Environment imports
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.vec_env import ParallelEnv
from torchrl.envs.transforms import RewardSum

# Model imports
from torchrl.objectives.utils import distance_loss
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules.models import MLP
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules import SafeModule, ProbabilisticActor, ValueOperator, ActorValueOperator

# Collector imports
from torchrl.collectors.collectors import SyncDataCollector

# Loss imports
from torchrl.objectives.value.advantages import GAE
from torchrl.objectives import ClipPPOLoss

# Training loop imports
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def main():

    args = get_args()

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    # 1. Define environment --------------------------------------------------------------------------------------------

    # 1.1 Define env factory
    def env_factory(num_workers):
        """Creates an instance of the environment."""

        create_env_fn = lambda: GymEnv(env_name="CartPole-v0")

        # 1.2 Create env vector
        vec_env = ParallelEnv(create_env_fn=create_env_fn, num_workers=num_workers)

        # 1.3 Apply transformations to vec env - standard DeepMind Atari - Order of transforms is important!
        transformed_vec_env = TransformedEnv(vec_env)
        transformed_vec_env.append_transform(RewardSum())

        return transformed_vec_env

    # Sanity check
    test_env = env_factory(num_workers=1)
    test_input = test_env.reset()
    num_inputs = test_env.specs["observation_spec"]["observation"].shape[-1]
    num_actions = test_env.specs["action_spec"].space.n

    # 2. Define model --------------------------------------------------------------------------------------------------

    # 2.1 Define input keys
    in_keys = ["observation"]

    # 2.2 Define shared module
    shared_mlp = MLP(
        in_features=num_inputs,
        activation_class=torch.nn.Tanh,
        activate_last_layer=True,
        out_features=128,
        num_cells=[128],
    )
    shared_module = SafeModule(  # Like TensorDictModule
        module=shared_mlp,
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # 2.2 Define actor

    # Define MLP net
    policy_mlp = MLP(
        in_features=128,
        activation_class=torch.nn.Tanh,
        activate_last_layer=False,
        out_features=num_actions,
        num_cells=[64],
    )

    # Define policy TensorDictModule
    policy_module = SafeModule(  # Like TensorDictModule
        module=policy_mlp,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        return_log_prob=True,
    )

    # 2.3 Define critic

    # Define MLP net
    value_mlp = MLP(
        in_features=128,
        activation_class=torch.nn.Tanh,
        activate_last_layer=False,
        out_features=1,
        num_cells=[64],
    )

    # Define TensorDictModule
    critic_module = ValueOperator(
        value_mlp,
        in_keys=["common_features"],
    )

    # Define actor critic

    actor_critic = ActorValueOperator(
        common_operator=shared_module,
        policy_operator=policy_module,
        value_operator=critic_module,
    ).to(device)

    # Get independent operators for actor and critic, to be able to call only one of them
    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    # 2. Define Collector ----------------------------------------------------------------------------------------------

    print(args.total_frames)
    collector = SyncDataCollector(
        create_env_fn=env_factory,
        create_env_kwargs={"num_workers": args.num_parallel_envs},
        policy=actor,
        total_frames=args.total_frames,
        device=device,
        passing_device=device,
        frames_per_batch=args.steps_per_env * args.num_parallel_envs,
    )

    # 3. Define Advantage module and loss ------------------------------------------------------------------------------

    advantage_module = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lamdda,
        value_network=critic,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=args.clip_epsilon,
        loss_critic_type=args.loss_critic_type,
        entropy_coef=args.entropy_coef,
        critic_coef=args.critic_coef,
        gamma=args.gamma,
    )

    # 4. Define logger -------------------------------------------------------------------------------------------------

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    # 5. Define training loop ------------------------------------------------------------------------------------------

    network_updates = 0
    collected_frames = 0
    batch_size = args.steps_per_env * args.num_parallel_envs
    num_mini_batches = batch_size // args.mini_batch_size
    optimizer = Adam(params=itertools.chain(actor.parameters(), critic.parameters()), lr=args.lr)
    total_network_updates = (args.total_frames // batch_size) * args.num_ppo_epochs * num_mini_batches
    scheduler = LinearLR(optimizer, total_iters=total_network_updates, start_factor=1.0, end_factor=0.1)

    with wandb.init(project=args.experiment_name, name=args.agent_name, config=args, mode=mode):

        start_time = time.time()

        for batch in collector:

            batch = batch.to(device)

            # Compute advantage with the whole batch
            with torch.no_grad():
                batch = advantage_module(batch)

            # We don't use memory networks, so sequence dimension is not relevant
            batch = batch[batch["mask"].squeeze(-1)]
            batch_size = batch.batch_size.numel()
            collected_frames += batch_size

            # add episode reward info
            train_episode_reward = batch["episode_reward"][batch["done"]]

            # PPO epochs
            for epoch in range(args.num_ppo_epochs):

                # Create a random permutation in every epoch
                for mini_batch_idxs in BatchSampler(
                        SubsetRandomSampler(range(batch_size)), args.mini_batch_size, drop_last=True):

                    log_info = {}

                    if batch["episode_reward"][batch["done"]].numel() > 0:
                        log_info.update({
                            "train_episode_rewards": train_episode_reward.mean(),
                        })

                    # Get data
                    mini_batch = batch[mini_batch_idxs].clone()

                    # Compute loss
                    loss = loss_module(mini_batch)
                    loss_sum = sum([item for key, item in loss.items() if key.startswith("loss")])

                    # Update networks
                    optimizer.zero_grad()
                    loss_sum.backward()

                    torch.nn.utils.clip_grad_norm_(itertools.chain(actor.parameters(), critic.parameters()), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()
                    network_updates += 1

                    log_info.update({
                        "loss": loss_sum.item(),
                        "loss_critic": loss["loss_critic"].item(),
                        "loss_objective": loss["loss_objective"].item(),
                        "lr": float(scheduler.get_last_lr()[0]),
                    })

                    if network_updates % args.evaluation_frequency == 0 and network_updates != 0:
                        # Run evaluation in test environment
                        with set_exploration_mode("random"):
                            test_env.eval()
                            test_td = test_env.rollout(
                                policy=actor,
                                max_steps=100_000,
                                auto_reset=True,
                                auto_cast_to_device=True,
                            ).clone()
                        log_info.update({"test_episode_reward": test_td["reward"].sum()})

                    # Print an informative message in the terminal
                    fps = int(collected_frames * args.frame_skip / (time.time() - start_time))
                    print_msg = f"Update {network_updates}, num " \
                                f"samples collected {collected_frames * args.frame_skip}, FPS {fps} "
                    for k, v in log_info.items():
                        print_msg += f"{k}: {v} "
                    print(print_msg, flush=True)

                    log_info.update({"collected_frames": int(collected_frames * args.frame_skip), "fps": fps})
                    wandb.log(log_info, step=network_updates)

                    del mini_batch

            # Update collector weights!
            collector.update_policy_weights_()


def get_args():
    """Reads conf.yaml file in the same directory"""

    parser = argparse.ArgumentParser(description='RL')

    # Configuration file, keep first
    parser.add_argument('--conf', '-c', default="conf.yaml")

    # Get args
    args = parser.parse_args()

    # Update args with cong.yaml
    if args.conf.endswith('yaml'):
        with open(args.conf) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        args.__dict__.update(config)

    return args


if __name__ == "__main__":
    main()

