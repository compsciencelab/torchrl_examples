import gc
import time
import yaml
import wandb
import tqdm
import torch
import argparse

# Environment imports
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.vec_env import ParallelEnv
from torchrl.envs.transforms import ToTensorImage, GrayScale, CatFrames, NoopResetEnv, Resize, ObservationNorm, RewardSum, StepCounter

# Model imports
from torchrl.envs import EnvCreator
from torchrl.envs.utils import set_exploration_mode, check_env_specs
from torchrl.modules.models import ConvNet, MLP
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules import SafeModule, ProbabilisticActor, ValueOperator, ActorValueOperator

# Collector imports
from torchrl.collectors.collectors import SyncDataCollector

# Loss imports
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

# Training loop imports
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.distributions as dist


def main():

    args = get_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device_collection = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 1. Define environment --------------------------------------------------------------------------------------------

    # 1.1 Define env factory
    def env_factory(device=device):
        """Creates an instance of the environment."""

        create_env_fn = EnvCreator(lambda: GymEnv(env_name=args.env_name, frame_skip=args.frame_skip, categorical_action_encoding=True))

        # 1.2 Create env vector
        vec_env = ParallelEnv(create_env_fn=create_env_fn, num_workers=args.num_parallel_envs)

        # 1.3 Apply transformations to vec env - standard DeepMind Atari - Order of transforms is important!
        transformed_vec_env = TransformedEnv(vec_env)
        # transformed_vec_env.append_transform(NoopResetEnv(noops=30))  # Start with 30 random action. incompatible with frame skip and ParallelEnv ?
        transformed_vec_env.append_transform(ToTensorImage())  # change shape from [h, w, 3] to [3, h, w]
        transformed_vec_env.append_transform(Resize(w=84, h=84))  # Resize image
        transformed_vec_env.append_transform(GrayScale())  # Convert to Grayscale
        transformed_vec_env.append_transform(ObservationNorm(in_keys=["pixels"], standard_normal=True))
        transformed_vec_env.append_transform(CatFrames(N=4))  # Stack last 4 frames
        transformed_vec_env.append_transform(RewardSum())
        transformed_vec_env.append_transform(StepCounter())

        norm_layer = transformed_vec_env.transform[3]
        norm_layer.init_stats(num_iter=1000, reduce_dim=[0, 1, 3, 4], cat_dim=0, keep_dims=[3, 4])

        return transformed_vec_env.to(device)

    # Sanity check
    test_env = env_factory()
    train_env = env_factory()
    check_env_specs(train_env)

    test_env.load_state_dict(train_env.state_dict())

    test_input = test_env.reset()
    assert "pixels" in test_input.keys()
    num_actions = test_env.specs["action_spec"].space.n

    # 2. Define model --------------------------------------------------------------------------------------------------

    # 2.1 Define input keys
    in_keys = ["pixels"]

    # 2.2 Define a shared Module and TensorDictModule (CNN + MLP)

    # Define CNN
    common_cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    ).to(device)
    common_cnn_output = common_cnn(torch.ones_like(test_input["pixels"]))

    # Add MLP on top of shared MLP
    common_mlp = MLP(
        in_features=common_cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=448,
        num_cells=[256]).to(device)
    common_mlp_output = common_mlp(common_cnn_output)

    # Define shared net as TensorDictModule
    common_module = SafeModule(  # Like TensorDictModule
        module=torch.nn.Sequential(common_cnn, common_mlp),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # 2.3 Define on head for the policy

    # Define Module
    policy_net = MLP(
        in_features=common_mlp_output.shape[-1],
        out_features=num_actions,
        num_cells=[]
    ).to(device)

    # Define TensorDictModule
    policy_module = SafeModule(  # TODO: The naming of SafeModule is confusing
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],  # TODO: Seems like only "logits" can be used as in_keys
        #        out_keys=["action"],
        distribution_class=dist.Categorical,
        distribution_kwargs={},
        return_log_prob=True,
    )

    # 2.4 Define another head for the value

    # Define Module
    value_net = MLP(
        in_features=common_mlp_output.shape[-1],
        out_features=1,
        num_cells=[]
    ).to(device)

    # Define TensorDictModule
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    # 2.5 Wrap modules in a single ActorCritic operator

    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    ).to(device)

    # 2.6 Initialize the model by running a forward pass
    with torch.no_grad():
        td = test_env.rollout(max_steps=1000)
        td = td.to(device)
        td = actor_critic(td)

    print(actor_critic)
    # TODO: why wrap them together and then get separate operators ?
    # Get independent operators for actor and critic, to be able to call only one of them
    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    # sanity check
    actor(test_env.reset())
    actor_critic(test_env.reset())

    # # Ugly hack, otherwise I get errors
    # critic.out_keys = ['state_value', 'common_features']
    # actor.out_keys = ['action', 'common_features', 'logits']

    # 2. Define Collector ----------------------------------------------------------------------------------------------

    collector = SyncDataCollector(
        create_env_fn=train_env,
        create_env_kwargs=None,
        policy=actor_critic,
        total_frames=args.total_frames,
        frames_per_batch=args.steps_per_env * args.num_parallel_envs,
        device=device_collection,
        passing_device=device_collection,
    )

    # 3. Define Loss ---------------------------------------------------------------------------------------------------

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
        mode = "offline"

    # 5. Define training loop ------------------------------------------------------------------------------------------

    network_updates = 0
    collected_frames = 0
    batch_size = args.steps_per_env * args.num_parallel_envs
    num_mini_batches = batch_size // args.mini_batch_size
    total_network_updates = (args.total_frames // batch_size) * args.num_ppo_epochs * num_mini_batches
    optimizer = Adam(params=actor_critic.parameters(), lr=args.lr)
    scheduler = LinearLR(optimizer, total_iters=total_network_updates, start_factor=1.0, end_factor=0.1)
    evaluation_frequency = 100  # In number of network frames

    with wandb.init(project=args.experiment_name, name=args.agent_name, entity=args.entity, config=args, mode=mode):

        pbar = tqdm.tqdm(total=collector.total_frames)
        start_time = time.time()

        for batch in collector:
            batch = batch.cpu()
            log_info = {}

            # We don't use memory networks, so sequence dimension is not relevant
            batch_size = batch["mask"].sum().item()
            pbar.update(batch_size)
            collected_frames += batch_size

            # add episode reward info
            # train_episode_reward = batch["episode_reward"][batch["done"]]
            #if batch["episode_reward"][batch["done"]].numel() > 0:
            #     log_info.update({"train_episode_rewards": train_episode_reward.mean()})
            # episode_steps = batch["episode_steps"][batch["done"]]

            # PPO epochs
            for epoch in range(args.num_ppo_epochs):

                # Compute advantage with the whole batch
                batch = advantage_module(batch.to(device))

                batch_view = batch[batch["mask"].squeeze(-1)]

                # Create a random permutation in every epoch
                for mini_batch_idxs in BatchSampler(
                        SubsetRandomSampler(range(batch_size)), args.mini_batch_size, drop_last=True):

                    # select idxs to create mini_batch
                    mini_batch = batch_view[mini_batch_idxs].clone().to(device)

                    # Forward pass
                    loss = loss_module(mini_batch)

                    # Add up losses
                    loss_sum = sum([item for key, item in loss.items() if key.startswith("loss")])

                    # Update networks
                    optimizer.zero_grad()
                    loss_sum.backward()
                    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()
                    network_updates += 1

                    log_info.update({
                        "loss": loss_sum.item(),
                        "loss_critic": loss["loss_critic"].item(),
                        "loss_entropy": loss["loss_entropy"].item(),
                        "loss_objective": loss["loss_objective"].item(),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "collected_frames": collected_frames,
                        "reward": batch["reward"][batch["mask"]].mean().item(),
                    })

                    if network_updates % args.evaluation_frequency == 0 and network_updates != 0:
                        # Run evaluation in test environment
                        with set_exploration_mode("random"):
                            test_env.eval()
                            test_td = test_env.rollout(
                                policy=actor,
                                max_steps=100000,
                                auto_reset=True,
                                auto_cast_to_device=True,
                            ).clone()
                        log_info.update({"test_reward": test_td["reward"].squeeze(-1).sum(-1).mean()})

                    # Print an informative message in the terminal
                    fps = int(collected_frames * args.frame_skip / (time.time() - start_time))
                    print_msg = f"Update {network_updates}, num " \
                                f"samples collected {collected_frames * args.frame_skip}, FPS {fps} "
                    for k, v in log_info.items():
                        print_msg += f"{k}: {v: 4.4f} "
                    pbar.set_description(print_msg)

                    log_info.update({"collected_frames": int(collected_frames * args.frame_skip), "fps": fps})
                    wandb.log(log_info, step=network_updates)
                    del mini_batch

            # Update collector weights!
            collector.update_policy_weights_()
            gc.collect()


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

