from typing import Dict, Tuple

import numpy as np

from cambrian.utils import safe_eval
from parse_evos import Data, Generation, Rank
from parse_types import AxisData, ParsedAxisData
from parse_helpers import get_axis_label


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
    pattern: str,
    convert_to_radians: bool = False,
) -> ParsedAxisData:
    p1, p2 = _eye_range_parse(rank_data, pattern=pattern)

    if convert_to_radians:
        p1 = np.deg2rad(p1)
        p2 = np.deg2rad(p2)
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