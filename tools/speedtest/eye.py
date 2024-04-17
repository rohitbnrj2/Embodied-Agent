from typing import List
import time

import mujoco as mj

from cambrian.animals import MjCambrianAnimal
from cambrian.eyes import MjCambrianEye
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import MjCambrianConfig, run_hydra


def main(config: MjCambrianConfig):
    xml = MjCambrianXML.from_string(config.env.xml)

    # NOTE: Only uses the first animal
    animal_config = next(iter(config.env.animals.values()))
    animal = MjCambrianAnimal(animal_config, "animal_0", 0)
    xml += animal.generate_xml()

    # Add the eyes
    eyes: List[MjCambrianEye] = []
    for name, eye_config in animal_config.eyes.items():
        eye = eye_config.instance(eye_config, name)
        xml += eye.generate_xml(xml, animal.geom, animal_config.body_name)
        eyes.append(eye)

    # Load the model and data
    model = mj.MjModel.from_xml_string(xml.to_string())
    data = mj.MjData(model)
    mj.mj_step(model, data)

    # Reset the eyes
    for eye in eyes:
        eye.reset(model, data)

    # Run the simulation
    print("Running simulation...")

    # Every 1 second, we'll print out the FPS of the simulation. Two values are
    # printed: time for all eyes to render and time per eye to render.
    t0 = time.time()
    window: int = 100
    for i in range(window * 10):
        for eye in eyes:
            eye.step()

        if i % window == 0 and i > 0:
            t1 = time.time()
            fps_all_eyes = window / (t1 - t0)
            fps_each_eye = fps_all_eyes * len(eyes)
            print(f"FPS: {fps_all_eyes:.2f} (all eyes), {fps_each_eye:.2f} (per eye)")
            print(
                f"Time: {1000 / fps_all_eyes:.2f} ms (all eyes), {1000 / fps_each_eye:.2f} ms (per eye)"
            )
            t0 = t1


if __name__ == "__main__":
    run_hydra(main)
