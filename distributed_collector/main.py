import ray
from fake_collector import FakeCollector
from distributed_collector import DistributedCollector


if __name__ == "__main__":

    # Init locally for now, but in a cluster is essentially the same with more params.
    ray.init()

    # Define resources of each remote collector
    default_remote_config = {
        "num_cpus": 1,
        "num_gpus": 0.2,
        "memory": 5 * 1024 ** 3,
        "object_store_memory": 2 * 1024 ** 3
    }

    print("\nTest 1: Collect data test in synchronous mode.\n")

    distributed_collector = DistributedCollector(
        collector_class=FakeCollector,
        collector_params={
            "num_batches": 100,
            "shape": (2, 10),
        },
        remote_config=default_remote_config,
        num_collectors=3,
        total_frames=1000,
        communication="sync",
    )

    counter = 0
    for batch in distributed_collector:
        counter += 1
        print(f"batch {counter}, shape {batch.shape}")
    distributed_collector.stop()

    print("\nTest 2: Collect data test in asynchronous mode.\n")

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

    print("\nSuccess!")
