import matplotlib.pyplot as plt
import numpy as np


from parse_types import ExtractedData


def extract_data(
    ax: plt.Axes,
    *,
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

        return_values.append(c_data)

    if return_size:
        s_data = []
        for collection in ax.collections:
            s = collection.get_sizes()
            s_data.append(s)
        s_data = np.array(s_data)

        return_values.append(s_data)

    return tuple(return_values) if len(return_values) > 1 else return_values[0]
