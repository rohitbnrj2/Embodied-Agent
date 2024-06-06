# custom_fn: Callable[[AxisData, Data, Generation, Rank], ParsedAxisData]
# ParsedAxisData: Tuple[np.ndarray | float | int, str]

from typing import Dict, Tuple

from cambrian.utils import safe_eval
from cambrian.utils.config.utils import glob
from parse_evos import AxisData, ParsedAxisData, Data, Generation, Rank, get_axis_label


# ==================
# Eye range parsing

def _eye_range_parse(rank_data: Rank, *, pattern: str) -> Tuple[float, float]:
    assert rank_data.config is not None, "Rank data config is required to plot eye placement."

    # Get the eye placement range from the globbed data
    key = pattern.split(".")[-1]
    eye_range = rank_data.config.globbed_eval(key, key=key, **{key: pattern})
    assert len(eye_range) == 1, "Only one animal is supported"
    eye_range = eye_range[0]
    assert len(eye_range) == 2, "Eye range must have 2 values."
    return eye_range

def eye_range_diff(
    axis_data: AxisData,
    data: Data,
    generation_data: Generation,
    rank_data: Rank,
    *,
    pattern: str
) -> ParsedAxisData:
    p1, p2 = _eye_range_parse(rank_data, pattern=pattern)

    return p2 - p1, get_axis_label(axis_data)

# ==================

def eval_safe(
    axis_data: AxisData,
    data: Data,
    generation_data: Generation,
    rank_data: Rank,
    *,
    src: str,
    patterns: Dict[str, str]
) -> ParsedAxisData:
    """This custom axis fn will pass a pattern directly to `safe_eval` using the 
    `rank_data.config`."""

    variables = {}
    for key, pattern in patterns.items():
        variable = rank_data.config.glob(pattern, flatten=True)
        assert len(variable) == 1, f"Pattern {pattern} must return a single value."
        variables[key] = next(iter(variable.values()))

    return safe_eval(src, variables), get_axis_label(axis_data)