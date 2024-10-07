import matplotlib.pyplot as plt
import numpy as np


from parse_types import ExtractedData, PlotData


def extract_data(
    ax: plt.Axes,
    *,
    plot_data: PlotData | None = None,
    return_data: bool = False,
    return_color: bool = False,
    return_size: bool = False,
) -> ExtractedData:
    """Extracts the data from a figure."""
    assert (
        return_data or return_color or return_size
    ), "At least one return value is required."
    return_values = []

    if return_data:
        x_data, y_data, z_data = [], [], []
        for collection in ax.collections:
            offset: np.ndarray = collection.get_offsets()

            if offset.shape[-1] == 2:
                x, y = offset.T
                x_data.append(x)
                y_data.append(y)
            else:
                x, y, z = offset.T
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
        try:
            x_data, y_data = np.array(x_data), np.array(y_data)
            z_data = None if not z_data else np.array(z_data)
        except ValueError:
            import pdb

            pdb.set_trace()

        if plot_data is not None:
            if plot_data.x_data.remove_outliers:
                x_data = remove_outliers(x_data)
            if plot_data.y_data.remove_outliers:
                y_data = remove_outliers(y_data)
            if (
                plot_data.z_data is not None
                and plot_data.z_data.remove_outliers
                and z_data is not None
            ):
                z_data = remove_outliers(z_data)

        return_values.extend([x_data, y_data, z_data])

    if return_color:
        c_data = []
        for collection in ax.collections:
            c = collection.get_array()
            c_data.append(c)
        c_data = np.array(c_data)

        # All the colors are None if the color is set to a solid color (i.e. SOLID)
        # We'll return None in that case
        if np.all([c is None for c in c_data]):
            c_data = None

        if plot_data is not None:
            if c_data is not None and plot_data.color_data.remove_outliers:
                import pdb

                pdb.set_trace()
                c_data = remove_outliers(c_data)

        return_values.append(c_data)

    if return_size:
        s_data = []
        for collection in ax.collections:
            s = collection.get_sizes()
            s_data.append(s)
        s_data = np.array(s_data)

        return_values.append(s_data)

    return tuple(return_values) if len(return_values) > 1 else return_values[0]


def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Replace outliers in a NxM array with the minimum value of each column.

    Parameters:
        data (np.ndarray): An NxM array representing RGB colors.
        threshold (float): The z-score threshold to identify outliers.

    Returns:
        np.ndarray: The array with outliers replaced by the minimum value in each column.
    """
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    mask = z_scores >= threshold
    min_values = np.min(data, axis=0)

    for i in range(data.shape[1]):
        data[mask[:, i], i] = min_values[i]

    return data
