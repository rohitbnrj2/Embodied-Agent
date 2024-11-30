from typing import Callable, Dict, List

import numpy as np
from parse_evos import Rank
from parse_helpers import get_size_label
from parse_types import ParsedAxisData, SizeData

from cambrian.utils import safe_eval


def select_rank(
    size_data: SizeData,
    rank_data: Rank,
    *,
    selection_fn: Callable[[np.ndarray], np.ndarray],
    per_generation: bool = False,
    n: int | slice = 1,
) -> ParsedAxisData:
    if isinstance(n, int):
        n = slice(None, n)

    def get_valid_ranks(ranks: List[Rank]):
        """Returns ranks with valid fitness."""
        return [r for r in ranks if r.eval_fitness is not None]

    def select_top_ranks(ranks: List[Rank]):
        """Selects top ranks based on the selection function."""
        fitness_values = np.array([r.eval_fitness for r in ranks])
        selected_fitness = selection_fn(fitness_values)
        if isinstance(selected_fitness, (int, float)):
            selected_fitness = [selected_fitness]
        return [r for r in ranks if r.eval_fitness in set(selected_fitness[n])]

    if per_generation:
        generations = rank_data.generation.data.generations.values()
        ranks = [get_valid_ranks(g.ranks.values()) for g in generations]
        selected_ranks = [select_top_ranks(gen_ranks) for gen_ranks in ranks]
        selected_ranks = [r for gen_ranks in selected_ranks for r in gen_ranks]
    else:
        ranks = get_valid_ranks(
            [
                r
                for g in rank_data.generation.data.generations.values()
                for r in g.ranks.values()
            ]
        )
        selected_ranks = select_top_ranks(ranks)

    if rank_data in selected_ranks:
        return size_data.size_max, "Selected Agent"
    else:
        return size_data.size_min, "Other Agents"


def num_children(size_data: SizeData, rank_data: Rank) -> ParsedAxisData:
    """Return the size of the rank_data based on the number of children it has."""
    return len(rank_data.children), "Number of Children"


def num_descendents(size_data: SizeData, rank_data: Rank) -> ParsedAxisData:
    """This is similar to num_children, but it counts all the children, grandchildren,
    etc."""

    def count_descendents(rank_data: Rank) -> int:
        count = len(rank_data.children)
        for child in rank_data.children:
            count += count_descendents(child)
        return count

    return count_descendents(rank_data), "Number of Descendents"


def eval_safe(
    size_data: SizeData,
    rank_data: Rank,
    *,
    src: str,
    patterns: Dict[str, str],
    assume_one: bool = True,
) -> ParsedAxisData:
    """This custom axis fn will pass a pattern directly to `safe_eval` using the
    `rank_data.config`."""

    variables = {}
    for key, pattern in patterns.items():
        variables[key] = rank_data.config.glob(
            pattern, flatten=True, assume_one=assume_one
        )

    try:
        return safe_eval(src, variables), get_size_label(size_data)
    except TypeError as e:
        raise TypeError(f"Failed to evaluate {src} with variables {variables}") from e
