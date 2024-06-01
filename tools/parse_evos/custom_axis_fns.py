# custom_fn: Callable[[AxisData, Data, Generation, Rank], ParsedAxisData]
# ParsedAxisData: Tuple[np.ndarray | float | int, str]

from parse_evos import AxisData, ParsedAxisData, Data, Generation, Rank


def parse_eye_range(
    axis_data: AxisData,
    data: Data,
    generation_data: Generation,
    rank_data: Rank,
    *,
    pattern: str
) -> ParsedAxisData:
    assert rank_data.config is not None, "Rank data config is required to plot eye placement."

    # Get the eye placement range from the globbed data
    globbed_data = rank_data.config.glob(pattern, flatten=True)
    eye_range_pattern = pattern.split(".")[-1]
    assert eye_range_pattern in globbed_data, f"Pattern {eye_range_pattern} not found."
    eye_range = globbed_data[eye_range_pattern]
    assert len(eye_range) == 2, "Eye range must have 2 values."
    p1, p2 = eye_range

    raise NotImplementedError("Eye range parsing is not implemented.")