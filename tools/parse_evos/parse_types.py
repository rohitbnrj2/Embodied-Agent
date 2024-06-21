from typing import List, Tuple, TypeAlias

import numpy as np

Color: TypeAlias = List[Tuple[float, float, float, float]] | List[float]
Size: TypeAlias = List[float]

ParsedAxisData: TypeAlias = Tuple[np.ndarray | float | int, str]
ParsedColorData: TypeAlias = Tuple[Color, str]
ParsedSizeData: TypeAlias = Tuple[Size, str]
ParsedPlotData: TypeAlias = Tuple[
    ParsedAxisData,
    ParsedAxisData,
    ParsedAxisData | None,
    ParsedColorData,
    ParsedSizeData,
]
ExtractedData: TypeAlias = Tuple[np.ndarray | None]