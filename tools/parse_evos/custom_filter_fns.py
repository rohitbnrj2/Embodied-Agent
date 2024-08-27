from typing import Dict, List, Tuple, Any

from parse_types import Data, Generation, Rank


def filter_noop(data: Data) -> Dict[int, Generation]:
    return data.generations


def filter_best_n_lineages(data: Data, *, n: int) -> Dict[int, Generation]:
    """Filter function to only keep the best n lineages.

    Args:
        data (Data): The data to filter.

    Keyword Args:
        n (int): The number of lineages to keep.

    Returns:
        Dict[int, Generation]: The filtered generations.
    """

    # First, we'll find the best n agents in the last generation
    sorted_generations: List[int] = sorted(list(data.generations.keys()))
    best_ranks: List[Rank] = sorted(
        list(data.generations[sorted_generations[-1]].ranks.values()),
        key=lambda x: x.eval_fitness,
        reverse=True,
    )[:n]

    # Now only return the ranks from all previous generations which this rank derives from
    for generation in sorted_generations[::-1]:
        for rank in data.generations[generation].ranks.values():
            if rank not in best_ranks:
                rank.ignored = True

        best_ranks = [rank.parent for rank in best_ranks]
        if any(rank is None for rank in best_ranks):
            break

    return data.generations


def filter_best_n_agents(data: Data, *, n: int) -> Dict[int, Generation]:
    """Filter the best n agents for each generation"""
    sorted_generations: List[int] = sorted(list(data.generations.keys()))
    for generation in sorted_generations[::-1]:
        best_ranks: List[Rank] = sorted(
            list(data.generations[generation].ranks.values()),
            key=lambda x: (
                x.eval_fitness if x.eval_fitness is not None else -float("inf")
            ),
            reverse=True,
        )[:n]

        for rank in data.generations[generation].ranks.values():
            if rank not in best_ranks:
                rank.ignored = True

    return data.generations


def filter_median_agent(data: Data) -> Dict[int, Generation]:
    """Filter the median agent for each generation"""
    sorted_generations: List[int] = sorted(list(data.generations.keys()))
    for generation in sorted_generations[::-1]:
        ranks = list(data.generations[generation].ranks.values())
        ranks.sort(
            key=lambda x: (
                x.eval_fitness if x.eval_fitness is not None else -float("inf")
            )
        )
        median_rank = ranks[len(ranks) // 2]
        for rank in data.generations[generation].ranks.values():
            if rank != median_rank:
                rank.ignored = True

    return data.generations


def filter_uniformly(
    data: Data, *, patterns: Dict[str, Tuple[int, int]]
) -> Dict[int, Generation]:
    """This filter will, for each pattern passed, will uniformly sample each on from the
    given range and find the agent with that config that's closest to it and return that
    agent."""

    num_generations = len(data.generations)
    generations: Dict[Generation, Dict[str, Any]] = {
        g: {p: (r[1] - r[0]) / num_generations * i + r[0] for p, r in patterns.items()}
        for i, g in enumerate(data.generations.values())
    }

    for generation, agent in generations.items():
        # Find the agent which is closests to all patterns
        best_agent = None
        best_distance = float("inf")
        for rank in generation.ranks.values():
            distance = sum(
                abs(
                    agent[pattern]
                    - rank.config.glob(pattern, flatten=True, assume_one=True)
                )
                for pattern in patterns.keys()
                if rank.config is not None
            )
            if distance < best_distance:
                best_distance = distance
                best_agent = rank

        for rank in generation.ranks.values():
            if rank != best_agent:
                rank.ignored = True

    return data.generations
