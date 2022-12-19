from typing import Optional, Sequence

import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase

# Environment imports
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.vec_env import ParallelEnv
from torchrl.envs.transforms import ToTensorImage, GrayScale, CatFrames, NoopResetEnv, Resize, ObservationNorm


class StepLimit(Transform):
    """Limits number of time steps in environments episodes.

    This transform keeps track of the total number of episode steps and sets ´done´ to True
    after max_episode_steps. If return_num_steps is True, adds the length of the episodes to
    to the input tensordict.
    """

    inplace = True

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        max_episode_steps: Optional[int] = None,
        return_num_steps: bool = False,
    ):
        in_keys = ["done"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.return_num_steps = return_num_steps
        self.max_episode_steps = max_episode_steps
        self.current_episode_steps = None  # Lazy init

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs):
        import ipdb; ipdb.set_trace()
        return tensordict

    # def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""
        self._check_inplace()

        # Get done tensor
        done = tensordict.get("done")
        not_done = 1 - done.to(torch.int32)

        # Update step count
        if self.current_episode_steps is None:
            self.current_episode_steps = torch.zeros_like(done).to(torch.int32)
        self.current_episode_steps += 1

        # Set updated done tensor
        if self.max_episode_steps is not None:
            done = torch.logical_or(done.to(torch.bool), self.current_episode_steps == self.max_episode_steps)
            tensordict.set("done", done)

        # Add episode_steps to tensordict if required
        if self.return_num_steps:
            tensordict.set("episode_steps",  self.current_episode_steps)

        # Reset step count if end of episode
        self.current_episode_steps *= not_done

        return tensordict

