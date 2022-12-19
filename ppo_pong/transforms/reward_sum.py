from typing import Optional, Sequence

import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase

# Environment imports
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.vec_env import ParallelEnv
from torchrl.envs.transforms import ToTensorImage, GrayScale, CatFrames, NoopResetEnv, Resize, ObservationNorm


class RewardSum(Transform):
    """Tracks the accumulated reward of each episode.

    This transform requires ´reward´ and ´done´ to be input keys. If that is not the case,
    the transform has no effect.
    """

    inplace = True

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ):
        in_keys = ["reward", "done"]
        if out_keys is None:
            out_keys = ["episode_reward"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.is_new_episode = None

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""
        self._check_inplace()

        for in_key in self.in_keys:
            if not in_key in tensordict.keys():
                return tensordict

        # Get input keys
        reward = tensordict.get("reward")
        done = tensordict.get("done")
        done = done.to(reward.dtype)

        # self.new_episode not initialized, assume its a new episode in all envs
        if self.is_new_episode is None:
            self.is_new_episode = torch.zeros_like(done)

        for out_key in self.out_keys:
            if not out_key in tensordict.keys():
                tensordict.set(out_key, torch.zeros(*tensordict.shape, 1, dtype=reward.dtype))
            updated_value = tensordict.get(out_key) * self.is_new_episode + reward
            tensordict.set(out_key,  updated_value)

        # Restart sum immediately after end-of-episode detected
        self.is_new_episode = 1 - done

        return tensordict


if __name__ == "__main__":

    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cuda:0")

    # 1. Define environment --------------------------------------------------------------------------------------------

    # 1.1 Define env factory
    def env_factory():
        """Creates an instance of the environment."""

        create_env_fn = lambda: GymEnv(env_name="PongNoFrameskip-v4", frame_skip=4)

        # 1.2 Create env vector
        vec_env = ParallelEnv(create_env_fn=create_env_fn, num_workers=4)

        # 1.3 Apply transformations to vec env - standard DeepMind Atari - Order of transforms is important!
        transformed_vec_env = TransformedEnv(vec_env)
        transformed_vec_env.append_transform(ToTensorImage())  # Change shape from [h, w, 3] to [3, h, w] for convs
        transformed_vec_env.append_transform(Resize(w=84, h=84))  # Resize image
        transformed_vec_env.append_transform(GrayScale())  # Convert to Grayscale
        transformed_vec_env.append_transform(
            ObservationNorm(loc=0.0, scale=1 / 255.))  # Divide pixels values by 255
        transformed_vec_env.append_transform(CatFrames(N=4))  # Stack last 4 frames
        transformed_vec_env.append_transform(RewardSum())  # Stack last 4 frames

        return transformed_vec_env

    # Sanity check
    test_env = env_factory()
    td = test_env.rollout(max_steps=1000)
