import ray
import torch
from abc import ABC
from typing import Iterator
from torch.utils.data import IterableDataset
from tensordict.tensordict import TensorDict, TensorDictBase
from fake_collector import FakeCollector

default_remote_config = {
    "num_cpus": 1,
    "num_gpus": 0.2,
    "memory": 5 * 1024 ** 3,
    "object_store_memory": 2 * 1024 ** 3
}


class DistributedCollector(IterableDataset, ABC):
    """
    Class to better handle the operations of ensembles of Collectors.

    Contains common functionality across all collectors sets.

    Parameters
    ----------
    collector_class : class
        A collector class.
    collector_params : dict
        Collector class kwargs.
    remote_config : dict
        Ray resource specs for the remote collectors.
    num_collectors: int
        Total number of collectors in the set (including local collector)
    """

    def __init__(self,
                 collector_class,
                 collector_params,
                 remote_config=default_remote_config,
                 num_collectors=1,
                 total_frames=1000,
                 communication="sync",  # "sync" or "async"
                 ):

        if communication not in ("sync", "async"):
            raise ValueError(f"Communication parameter in CollectorSet has to be sync or async.")

        self.collected_frames = 0
        self.total_frames = total_frames
        self.collector_class = collector_class
        self.collector_params = collector_params
        self.num_collectors = num_collectors
        self.remote_config = remote_config
        self.communication = communication

        # Create a local instance of the collector class
        # TODO: actually not used for now, but sometimes can be interesting to have a local
        #  copy of the collector
        self._local_collector = self._make_collector(
            self.collector_class, collector_params)

        # Create remote instances of the collector class
        self._remote_collectors = []
        if self.num_collectors > 1:
            self.add_collectors(self.num_collectors - 1, collector_params)

    @staticmethod
    def _make_collector(cls, collector_params):
        """Create a single collector instance."""
        w = cls(**collector_params)
        return w

    def add_collectors(self, num_collectors, collector_params):
        """Create and add a number of remote collectors to the set."""
        cls = self.collector_class.as_remote(**self.remote_config).remote
        self._remote_collectors.extend(
            [self._make_collector(cls, collector_params) for _ in range(num_collectors)])

    def local_collector(self):
        """Return local collector"""
        return self._local_collector

    def remote_collectors(self):
        """Returns list of remote collectors"""
        return self._remote_collectors

    def stop(self):
        """Stop all remote collectors"""
        for w in self.remote_collectors():
            w.__ray_terminate__.remote()

    def __iter__(self) -> Iterator[TensorDictBase]:
        if self.communication == "sync":
            return self.sync_iterator()
        else:
            return self.async_iterator()

    def sync_iterator(self) -> Iterator[TensorDictBase]:

        while self.collected_frames < self.total_frames:

            # Broadcast weights
            policy_weights = {}  # TODO. get latest weights
            latest_weights = ray.put(policy_weights)
            for e in self.remote_collectors():
                e.set_weights.remote(latest_weights)

            # Ask for batches to all remote workers.
            pending_samples = [e.rollout.remote() for e in self.remote_collectors()]

            # Wait for all rollouts
            samples_ready = []
            while len(samples_ready) < self.num_collectors - 1:
                samples_ready, samples_not_ready = ray.wait(pending_samples, num_returns=len(pending_samples), timeout=0.001)

            # Retrieve and concatenate Tensordicts
            out_td = []
            for r in pending_samples:
                rollouts = ray.get(r)
                ray.internal.free(r)
                out_td.append(rollouts)
            out_td = torch.cat(out_td)

            self.collected_frames += out_td.numel()

            yield out_td

    def async_iterator(self) -> Iterator[TensorDictBase]:

        pending_tasks = {}
        for w in self.remote_collectors():
            future = w.rollout.remote()
            pending_tasks[future] = w

        while self.collected_frames < self.total_frames:

            if not len(list(pending_tasks.keys())) == len(self.remote_collectors()):
                raise RuntimeError("Missing pending tasks, something went wrong")

            # Wait for first worker to finish
            wait_results = ray.wait(list(pending_tasks.keys()))
            future = wait_results[0][0]
            w = pending_tasks.pop(future)

            # Retrieve single rollouts
            out_td = ray.get(future)
            ray.internal.free(future)
            self.collected_frames += out_td.numel()

            # Update agent weights
            policy_weights = {}  # TODO. get latest weights
            latest_weights = ray.put(policy_weights)
            w.set_weights.remote(latest_weights)

            # Schedule a new collection task
            future = w.rollout.remote()
            pending_tasks[future] = w

            yield out_td


if __name__ == "__main__":

    ray.init()
    distributed_collector = DistributedCollector(
        collector_class=FakeCollector,
        collector_params={
            "num_batches": 100,
            "shape": (2, 10),
        },
        remote_config=default_remote_config,
        num_collectors=3,
        total_frames=1000,
        communication="async",
    )

    counter = 0
    for batch in distributed_collector:
        counter += 1
        print(f"batch {counter}, shape {batch.shape}")
    distributed_collector.stop()
