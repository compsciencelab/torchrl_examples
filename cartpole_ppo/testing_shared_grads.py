import math
import yaml
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
from torchrl.modules.models import MLP
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules import SafeModule, ProbabilisticActor, ValueOperator, ActorValueOperator

# Loss imports
from torchrl.objectives.value.advantages import GAE

# Training loop imports
from torch.optim import Adam


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
        out_features=64,
        num_cells=[64],
    )
    shared_module = SafeModule(  # Like TensorDictModule
        module=shared_mlp,
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # 2.3 Define actor head

    # Define MLP net
    policy_mlp = MLP(
        in_features=64,
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

    # 2.4 Define critic head

    # Define MLP net
    value_mlp = MLP(
        in_features=64,
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

    # 2.5 Define actor critic

    actor_critic = ActorValueOperator(
        common_operator=shared_module,
        policy_operator=policy_module,
        value_operator=critic_module,
    ).to(device)

    # Get independent operators for actor and critic, to be able to call only one of them
    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    # 2.6 Initialize the model by running a forward pass
    with torch.no_grad():
        td = test_env.rollout(max_steps=100)
        td = td.to(device)
        td = actor(td)
        td = critic(td)
        del td

    # 3. Define adv and loss module  -----------------------------------------------------------------------------------

    advantage_module = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lamdda,
        value_network=critic,
        average_gae=True,
    )

    def loss_module(batch):

        # Get data
        log_clip_bounds = (math.log1p(-args.clip_epsilon), math.log1p(args.clip_epsilon))
        action = batch.get("action")
        advantage = batch.get("advantage")
        dist = actor.get_dist(batch.clone(recurse=False))
        log_prob = dist.log_prob(action)
        prev_log_prob = batch.get("sample_log_prob")

        # Actor loss
        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        gain1 = log_weight.exp() * advantage
        log_weight_clip = log_weight.clamp(*log_clip_bounds)
        gain2 = log_weight_clip.exp() * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        action_loss = -gain.mean()

        # Critic loss
        critic(batch)
        value = batch.get("state_value")
        loss_value = distance_loss(
            value,
            batch["value_target"],
            loss_function="l2",
        ).mean()

        # Entropy loss
        loss_entropy = dist.entropy().mean() * args.entropy_coef

        return action_loss, loss_value, loss_entropy

    # 5. Run checks ----------------------------------------------------------------------------------------------------

    optimizer = Adam(params=itertools.chain(actor.parameters(), critic.parameters()), lr=args.lr)

    with torch.no_grad():
        test_batch = test_env.rollout(max_steps=1000)
        test_batch = test_batch.to(device)
        test_batch = actor(test_batch)
        test_batch = advantage_module(test_batch)

    action_loss, value_loss, entropy_loss = loss_module(test_batch)

    # Backprop action
    action_loss.backward()

    # Check 1: all actor params should have weights different than zero
    #  Also, all critic ONLY parameters should be None
    sum_actor_params_1 = 0.0
    sum_critic_params_1 = 0.0
    sum_shared_params_1 = 0.0

    for name, param in actor_critic.named_parameters():
        if name.startswith("module.0"):  # shared module
            sum_shared_params_1 += param.grad.sum()
            assert torch.tensor(param.grad != 0.0).any()
        elif name.startswith("module.1"):  # actor module
            sum_actor_params_1 += param.grad.sum()
        elif name.startswith("module.2"):  # critic module
            assert param.grad is None

    # Backprop value
    value_loss.backward()

    # Check 2: critic ONLY parameters should not be None anymore,
    #  actor parameters should remain the same and shared parameters should change
    sum_actor_params_2 = 0.0
    sum_critic_params_2 = 0.0
    sum_shared_params_2 = 0.0
    for name, param in actor_critic.named_parameters():
        if name.startswith("module.0"):
            sum_shared_params_2 += param.grad.sum()
        elif name.startswith("module.1"):
            sum_actor_params_2 += param.grad.sum()
        elif name.startswith("module.2"):
            assert torch.tensor(param.grad != 0.0).any()
            sum_critic_params_2 += param.grad.sum()
            assert param.grad is not None
    assert sum_actor_params_1 == sum_actor_params_2
    assert sum_shared_params_1 != sum_shared_params_2

    # Check 3: if we update the losses in inverse order grads should be the same
    optimizer.zero_grad()
    action_loss, value_loss, entropy_loss = loss_module(test_batch)
    value_loss.backward()

    sum_actor_params_3 = 0.0
    sum_critic_params_3 = 0.0
    sum_shared_params_3 = 0.0
    for name, param in actor_critic.named_parameters():
        if name.startswith("module.0"):
            sum_shared_params_3 += param.grad.sum()
            assert torch.tensor(param.grad != 0.0).any()
        elif name.startswith("module.1"):
            assert param.grad.sum() == 0.0
        elif name.startswith("module.2"):
            sum_critic_params_3 += param.grad.sum()

    action_loss.backward()
    sum_actor_params_4 = 0.0
    sum_critic_params_4 = 0.0
    sum_shared_params_4 = 0.0
    for name, param in actor_critic.named_parameters():
        if name.startswith("module.0"):
            sum_shared_params_4 += param.grad.sum()
        elif name.startswith("module.1"):
            sum_actor_params_4 += param.grad.sum()
        elif name.startswith("module.2"):
            assert torch.tensor(param.grad != 0.0).any()
            sum_critic_params_4 += param.grad.sum()
            assert param.grad is not None
    assert sum_actor_params_2 == sum_actor_params_4
    assert sum_shared_params_2 == sum_shared_params_4
    assert sum_critic_params_2 == sum_critic_params_4

    print("Success!!")


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

