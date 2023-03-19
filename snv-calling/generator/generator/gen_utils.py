from typing import Callable, List, Set, TypeVar

from generator.core.types import CNEvent

T = TypeVar("T")


def cn_events_overlap(ev1: CNEvent, ev2: CNEvent) -> bool:
    return (ev1[0] <= ev2[0] < ev1[1]) or (ev2[0] <= ev1[0] < ev2[1])


def sample_conditionally(sampler: Callable[[], T], condition: Callable[[T], bool]) -> T:
    """
    Returns first result of call to @sampler which satisfies @condition
    """
    sample = sampler()
    while not condition(sample):
        sample = sampler()
    return sample


def sample_conditionally_without_replacement(
    k: int, sampler: Callable[[], T], condition: Callable[[T], bool]
) -> Set[T]:
    result = set()
    for _ in range(0, k):
        result.add(
            sample_conditionally(sampler, lambda x: x not in result and condition(x))
        )
    return result


def sample_conditionally_with_replacement(
    k: int, sampler: Callable[[], T], condition: Callable[[T], bool]
) -> List[T]:
    return [sample_conditionally(sampler, condition) for _ in range(0, k)]
