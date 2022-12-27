from typing import Optional, Sequence

import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase


class StepLimit(Transform):
    """Limits number of time steps in environments episodes.

    This transform keeps track of the total number of episode steps and sets ´done´ to True
    after max_episode_steps. It also adds the length of the episodes to the input tensordict.
    """

    inplace = True

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        max_episode_steps: Optional[int] = None,
    ):
        in_keys = ["done"]
        if out_keys is None:
            out_keys = ["episode_steps"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.max_episode_steps = max_episode_steps

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets episode rewards."""
        if "reset_workers" in tensordict.keys():
            for out_key in self.out_keys:
                tensordict[out_key][tensordict["reset_workers"]] = 0.0

        return tensordict

    # def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""

        # Sanity checks
        self._check_inplace()
        for in_key in self.in_keys:
            if in_key not in tensordict.keys():
                return tensordict

        # Get done tensor
        done = tensordict.get("done")
        not_done = 1 - done.to(torch.int32)
        for out_key in self.out_keys:
            if out_key not in tensordict.keys():
                tensordict.set(
                    out_key, torch.zeros(*tensordict.shape, 1, dtype=torch.int32)
                )
            tensordict[out_key] += 1 * not_done

            # If maximum number of step reached, set done frag to True
            if self.max_episode_steps is not None:
                done[tensordict[out_key] == self.max_episode_steps] = True
                tensordict.set("done", done)

        return tensordict

