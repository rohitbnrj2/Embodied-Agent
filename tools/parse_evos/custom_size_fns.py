from parse_evos import Rank
from parse_types import SizeData, ParsedAxisData

def best_n(
    size_data: SizeData,
    rank_data: Rank,
    *,
    n: int,
) -> ParsedAxisData:
    """If the rank is one of the best n, return size_data.size_max, else return 
    size_data.size_min. Will sort the ranks by the eval_fitness."""

    # Get the best n ranks
    ranks = [r for g in rank_data.generation.data.generations.values() for r in g.ranks.values() if r.eval_fitness is not None]
    best_n_ranks = sorted(ranks, key=lambda r: r.eval_fitness, reverse=True)[:n]

    # Check if the rank is in the best n
    if rank_data in best_n_ranks:
        return size_data.size_max, f"Top {n} Performing Agent"
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
