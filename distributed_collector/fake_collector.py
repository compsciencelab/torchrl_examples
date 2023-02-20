from abc import ABC
import ray
import torch
from torch.utils.data import IterableDataset
from tensordict.tensordict import TensorDict, TensorDictBase


class FakeCollector(IterableDataset, ABC):

    def __init__(self, num_batches=100, shape=(2, 10)):
        super(FakeCollector).__init__()
        self.shape = shape
        self.num_batches = num_batches

    @classmethod
    def as_remote(cls,
                  num_cpus=None,
                  num_gpus=None,
                  memory=None,
                  object_store_memory=None,
                  resources=None):
        """
        Creates an instance of a remote ray FakeCollector.

        Parameters
        ----------
        num_cpus : int
            The quantity of CPU cores to reserve for this class.
        num_gpus  : float
            The quantity of GPUs to reserve for this class.
        memory : int
            The heap memory quota for this class (in bytes).
        object_store_memory : int
            The object store memory quota for this class (in bytes).
        resources: Dict[str, float]
            The default resources required by the class creation task.

        Returns
        -------
        w : FakeCollector
            A ray remote FakeCollector class.
        """
        w = ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources)(cls)
        w.is_remote = True
        return w

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        return TensorDict({"observation": torch.randn(self.shape)}, batch_size=self.shape)

    def set_weights(self, policy_weights={}):
        """Update the policy version with provided weights."""
        pass


if __name__ == "__main__":

    collector = FakeCollector()
    counter = 0
    for _ in range(10):
        counter += 1
        batch = collector.rollout()
        print(f"batch {counter}, shape {batch.shape}")
