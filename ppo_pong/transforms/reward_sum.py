from typing import Optional, Sequence

import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase


class RewardSum(Transform):
    """Tracks the accumulated reward of each episode.
    This transform requires ´reward´ to be input key. If that is not the case,
    the transform has no effect.
    """

    inplace = True

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ):
        in_keys = ["reward"]
        if out_keys is None:
            out_keys = ["episode_reward"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets episode rewards."""
        if "reset_workers" in tensordict.keys():
            for out_key in self.out_keys:
                tensordict[out_key][tensordict["reset_workers"]] = 0.0

        return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Updates the episode rewards with the step rewards."""
        # Sanity checks
        self._check_inplace()
        for in_key in self.in_keys:
            if in_key not in tensordict.keys():
                return tensordict

        # Update episode rewards
        reward = tensordict.get("reward")
        for out_key in self.out_keys:
            if out_key not in tensordict.keys():
                tensordict.set(
                    out_key, torch.zeros(*tensordict.shape, 1, dtype=reward.dtype)
                )
            tensordict[out_key] += reward

        return tensordict
