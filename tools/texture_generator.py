from typing import Tuple

import cv2
import numpy as np

from cambrian.utils.logger import get_logger


def generate(args):
    import subprocess

    args.generate = False

    num_repeats = [2, 4, 6, 8, 10, 20, 30, 40, 60]
    for prefix in ["horizontal", "vertical", "checkered"]:
        for n in num_repeats:
            args.num_repeats = n
            args.horizontal = prefix == "horizontal"
            args.checkered = prefix == "checkered"

            filename = f"{prefix}_{n}.png"
            args.save = f"models/assets/maze_textures/{filename}"

            cmd = ["python", __file__]
            for k, v in vars(args).items():
                k = k.replace("_", "-")
                if isinstance(v, bool):
                    cmd += [f"--{k}"] if v else []
                elif isinstance(v, (tuple, list)):
                    cmd += [f"--{k} {' '.join(map(str, v))}"]
                else:
                    cmd += [f"--{k} {v}"]

            cmd = " ".join(cmd)
            get_logger().info(f"Running {cmd}")
            subprocess.run(cmd, shell=True)


def generate_texture(
    shape: Tuple[int, int], num_repeats: int, checkered: bool
) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)

    if num_repeats == 0:
        pass
    elif checkered:
        assert shape[0] % num_repeats == 0 and shape[1] % num_repeats == 0
        bar_height = shape[0] // num_repeats
        bar_width = shape[1] // num_repeats
        # Creating the checkerboard pattern
        for i in range(num_repeats):
            for j in range(num_repeats):
                if (i + j) % 2 == 0:
                    img[
                        i * bar_height : (i + 1) * bar_height,
                        j * bar_width : (j + 1) * bar_width,
                    ] = 255
    else:
        assert shape[1] % num_repeats == 0
        bar_width = shape[1] // num_repeats
        for i in range(num_repeats):
            if i % 2 == 0:
                img[:, i * bar_width : (i + 1) * bar_width] = 255

    # add rgba channels
    img = np.stack([img] * 4, axis=-1)

    return img


def generate_cube_texture(
    shape: Tuple[int, int],
    num_repeats: int,
    checkered: bool,
    horizontal: bool,
    side_only: bool,
    custom_file: str = None,
) -> np.ndarray:
    # (row, col): is_horizontal
    texture_map = {
        (0, 1): horizontal,
        (1, 0): not horizontal,
        (1, 1): not horizontal,
        (1, 2): not horizontal,
        (1, 3): not horizontal,
        (2, 1): horizontal,
    }

    images = []
    for row in range(3):
        image_row = []
        for col in range(4):
            is_horizontal_or_none = texture_map.get((row, col), None)
            if side_only:
                raise NotImplementedError("side_only not implemented")
            if is_horizontal_or_none is None:
                img = generate_texture(shape, 0, checkered)
            else:
                if custom_file:
                    # read as rgba
                    img = cv2.imread(custom_file)
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                    img = cv2.resize(img, shape)
                else:
                    img = generate_texture(shape, num_repeats, checkered)
                    if is_horizontal_or_none:
                        img = np.rot90(img)

            image_row.append(img)
        images.append(image_row)

    return np.vstack([np.hstack(image_row) for image_row in images])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--generate", action="store_true")

    parser.add_argument("-s", "--shape", type=int, nargs=2, default=(120, 120))
    parser.add_argument("-n", "--num-repeats", type=int, default=10)
    parser.add_argument("--horizontal", action="store_true")
    parser.add_argument(
        "--transparent", action="store_true", help="Converts black to transparent"
    )
    parser.add_argument("--custom-file", type=str)
    parser.add_argument("--checkered", action="store_true")
    parser.add_argument("--side-only", action="store_true")
    parser.add_argument("--cube", action="store_true")
    parser.add_argument("--save", type=str)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    assert args.num_repeats % 2 == 0, "Number of repeats must be even"

    if args.generate:
        generate(args)
        exit()

    if args.cube:
        img = generate_cube_texture(
            args.shape,
            args.num_repeats,
            args.checkered,
            args.horizontal,
            args.side_only,
            args.custom_file,
        )
    else:
        img = generate_texture(args.shape, args.num_repeats, args.checkered)
        if args.horizontal:
            img = np.rot90(img)

    if args.transparent:
        img[img[:, :, 0] == 0] = [0, 0, 0, 0]

    if args.save:
        cv2.imwrite(args.save, img)

    if args.show:
        import matplotlib.pyplot as plt

        plt.imshow(img, cmap="gray")
        plt.show()
