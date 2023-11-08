from typing import Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import glob

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy

from env import MjCambrianEnv


class PlotEvaluationCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the evaluations.npz file produced by the EvalCallback and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/evaluations.npz`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/monitor.png`.

    Keyword Args:
        verbose (int): The verbosity level. Defaults to 0.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str, *, verbose: int = 0):
        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.verbose = verbose
        self.n_calls = 0

    def _on_step(self) -> bool:
        if self.verbose > 0:
            print(f"Plotting evaluation results at {self.evaldir}")

        # Also save the monitor
        try:
            x, y = ts2xy(load_results(self.logdir), "timesteps")

            def moving_average(values, window):
                weights = np.repeat(1.0, window) / window
                return np.convolve(values, weights, "valid")

            y = moving_average(y.astype(float), window=min(len(y) // 10, 1000))
            x = x[len(x) - len(y) :]  # truncate x

            plt.plot(x, y)
            plt.xlabel("Number of Timesteps")
            plt.ylabel("Rewards")
            plt.savefig(self.evaldir / "monitor.png")
            plt.cla()
        except Exception as e:
            print(f"Couldn't save monitor: {e}.")

        return True


class SaveVideoCallback(BaseCallback):
    """Should be used with an EvalCallback to visualize the environment.

    This callback will save a visualization of the environment at the end of each
    evaluation. Should be passed as the `callback_after_eval` for the EvalCallback.

    NOTE: Only the first environment is visualized

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        max_episode_steps: int,
        *,
        composite_image_shape: Tuple[int, int] = (150, 150),
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        # Delete all the existing gifs
        for f in glob.glob(str(self.evaldir / "vis_*.gif")):
            if self.verbose > 0:
                print(f"Deleting {f}")
            Path(f).unlink()

        self.composite_image_shape = composite_image_shape

        self.max_episode_steps = max_episode_steps

    def _on_step(self) -> bool:
        """Set the camera position to render the images at."""
        assert self.parent.eval_env.num_envs == 1, "Only one env can be visualized."
        env: MjCambrianEnv = self.parent.eval_env.envs[0]
        assert env.render_mode == "rgb_array", "Env must be in rgb_array mode."

        # Update the viewer camera to be a birdseye view
        if env.mujoco_renderer.viewer is None:
            env.render()
        env.mujoco_renderer.viewer.cam.fixedcamid = -1
        env.mujoco_renderer.viewer.cam.distance = 30
        env.mujoco_renderer.viewer.cam.azimuth = 90
        env.mujoco_renderer.viewer.cam.elevation = -90
        env.mujoco_renderer.viewer.cam.lookat = np.array([0, 0, 0])

        images = []
        step_count = 0
        obs, _ = env.reset()
        while step_count < self.max_episode_steps:
            action, _ = self.parent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            image = env.render().copy()
            image = self._add_text(image, step_count)
            composite_image = self._create_composite_image(env, image.shape)
            image = np.vstack((image, composite_image))
            images.append(image)

            step_count += 1

        filename = f"vis_{self.n_calls}.gif"
        duration = 1000 * 1 / env.metadata["render_fps"]
        if self.verbose > 0:
            print(f"Saving visualization at {self.evaldir / filename}")
        imageio.mimsave(self.evaldir / filename, images, loop=0, duration=duration)

        return True

    def _create_composite_image(self, env: MjCambrianEnv, image_shape: Tuple[int, int]):
        """
        For each animal, we'll visualize their composite images along the right
        side of the image. We'll first initialize a blank image to concat to the
        right of the main image, and we'll overwrite it with each animal's
        composite image that is scaled to be the same for each animal.
        Each animal composite image will be scaled to have a width of 50 pixels
        and then cropped to be square.
        """
        assert min(self.composite_image_shape) > 50, "Composite image too small."
        num_images = len(env.animals)
        w, h = self.composite_image_shape
        num_columns = int(np.ceil(num_images / (image_shape[1] // h)))
        images_per_column = int(num_images // num_columns)
        full_composite_image = np.zeros(
            (num_columns * 2 * w, image_shape[1], 3), dtype=np.uint8
        )

        for i, (name, animal) in enumerate(env.animals.items()):
            i *= 2
            wi, hi = (i % images_per_column) * w, (i // images_per_column) * h
            start_w, end_w = wi * w, (wi + 1) * w
            start_h, end_h = hi * h, (hi + 1) * h

            composite_image = animal.create_composite_image()
            if composite_image is not None:
                new_composite_image = self._resize_composite_image(
                    composite_image, name
                )

                new_composite_image = self._put_text(
                    new_composite_image,
                    f"Num: {len(animal.eyes)}",
                    org=(3, 10),
                    fontScale=0.25,
                    thickness=1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                )
                new_composite_image = self._put_text(
                    new_composite_image,
                    f"Res: {next(iter(animal.eyes.values())).resolution}",
                    org=(3, 20),
                    fontScale=0.25,
                    thickness=1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                )
                new_composite_image = self._put_text(
                    new_composite_image,
                    f"FOV: {next(iter(animal.eyes.values())).fov}",
                    org=(3, 30),
                    fontScale=0.25,
                    thickness=1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                )

                full_composite_image[start_w:end_w, start_h:end_h] = new_composite_image

            intensity_sensor_image = np.rot90(animal.intensity_sensor.last_obs, -1)
            new_intensity_sensor_image = self._resize_composite_image(
                intensity_sensor_image, animal.intensity_sensor.name
            )
            full_composite_image[
                start_w + w : end_w + w, start_h:end_h
            ] = new_intensity_sensor_image

        return full_composite_image

    def _resize_composite_image(self, composite_image: np.ndarray, name: str):
        w, h = self.composite_image_shape
        assert w == h, "Composite image must be square."
        size = max(w, h)
        aspect = composite_image.shape[0] / composite_image.shape[1]
        if composite_image.shape[0] > composite_image.shape[1]:
            new_shape = (
                int(size / aspect),
                size,
            )
        else:
            new_shape = (
                size,
                int(size * aspect),
            )

        composite_image = cv2.resize(
            composite_image, new_shape, interpolation=cv2.INTER_AREA
        )

        new_composite_image = np.zeros((w, h, 3), dtype=np.uint8)

        new_composite_image[
            new_composite_image.shape[0] // 2
            - composite_image.shape[0] // 2 : new_composite_image.shape[0] // 2
            + composite_image.shape[0] // 2 + 1,
            new_composite_image.shape[1] // 2
            - composite_image.shape[1] // 2 : new_composite_image.shape[1] // 2
            + composite_image.shape[1] // 2 + 1,
        ] = composite_image
        new_composite_image = self._put_text(
            new_composite_image,
            f"{name}",
            org=(w // 100, h - h // 20),
            fontScale=0.23,
            thickness=1,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        )

        return new_composite_image

    def _add_text(self, image: np.ndarray, step: int) -> np.ndarray:
        image = self._put_text(image, f"Step: {step}", org=(5, 20))
        image = self._put_text(
            image,
            f"Best Mean Reward: {self.parent.best_mean_reward:.2f}",
            org=(5, 40),
        )
        image = self._put_text(
            image, f"Total Timesteps: {self.num_timesteps}", org=(5, 60)
        )
        return image

    def _put_text(
        self, image: np.ndarray, text: str, org: Tuple[int, int], **kwargs
    ) -> np.ndarray:
        if "fontFace" not in kwargs:
            kwargs["fontFace"] = 1
        if "fontScale" not in kwargs:
            kwargs["fontScale"] = 1
        if "color" not in kwargs:
            kwargs["color"] = (255, 255, 255)
        if "thickness" not in kwargs:
            kwargs["thickness"] = 1
        return cv2.putText(
            image,
            text,
            org=org,
            **kwargs,
        )


class CallbackListWithSharedParent(CallbackList):
    def __init__(self, *args, **kwargs):
        self.callbacks = []
        super().__init__(*args, **kwargs)

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent
