from typing import List
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, flag_override
import matplotlib.pyplot as plt

from cambrian.utils.base_config import MjCambrianDictConfig
from cambrian.utils.config import MjCambrianConfig, config_wrapper

@config_wrapper
class Data:
    path: Path
    config: MjCambrianConfig
    return_value: float

def get_sequential_experiment_folders(directory: Path) -> List[Path]:
    folders: List[Path] = []
    for folder in directory.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            folders.append(folder)
    folders.sort(key=lambda x: int(x.name))
    return folders

def find_job_end_callback_return_value_from_file(file_path):
    with open(file_path, 'r') as file:
        log_contents = file.read()
    for line in reversed(log_contents.splitlines()):
        if "callbacks.on_job_end :: Succeeded with return value:" in line:
            return float(line.split(":")[-1].strip())
    return None


def load_data(paths: List[Path]) -> List[Data]:
    data = []
    for path in paths:
        print(path)
        with hydra.initialize_config_dir(str(path.absolute()), version_base=None):
            config = hydra.compose(config_name="config", return_hydra_config=True)
            HydraConfig.instance().set_config(config)
            OmegaConf.set_struct(config, False)
            del config["hydra"]
            config = MjCambrianConfig.instantiate(config)
        # config = MjCambrianDictConfig(MjCambrianConfig.load(path / "config.yaml", instantiate=False))
        return_value = find_job_end_callback_return_value_from_file(path / "logs" / "out.log")
        if return_value is not None:
            data.append(Data(path=path, config=config, return_value=return_value))
        
    return data

def plot_data(data: List[Data], key: str):
    # Plots the specific key vs the return value
    f = plt.figure()
    for d in data:
        plt.plot(d.config.select(key, throw_on_missing=True), d.return_value, 'ro')

    # Save the plot
    f.savefig(f"{key}.png")

def main(folder: Path):
    # with hydra.initialize(version_base=None):
    paths = get_sequential_experiment_folders(folder)
    data = load_data(paths)

    f = plt.figure()
    for d in data:
        num_eyes = len([eye for eye in d.config.env.animals["animal_0"].eyes.values() if eye.enabled])
        plt.plot(num_eyes, d.return_value, 'ro')
    
    f.savefig("plot.png")


    # result = data[0].config.interpolate("${eval:'([i for i in ${oc.dict.values:env.animals.animal_0.eyes}])'}")
    # print(OmegaConf.get_type(result))
    # print(type(result))
    # exit()
    # temp = OmegaConf.create(dict(custom="${logdir}"))
    # data[0].config.merge_with(temp)
    # data[0].config.resolve()
    # print(data[0].config.custom)
    # # OmegaConf.merge(data[0].config, temp)
    # exit()
    # print((data[0].config.merge_with(OmegaConf.create(dict(expname="${logdir}")))).expname)
    # print(data[0].config.select("${eval:'len([i for i in ${oc.dict.values:env.animals.animal_0.eyes} if i.enabled])}"))

    # plot_data(data, "env.animals.animal_0.eyes.eye_0.resolution.0")
    # plot_data(data, "env.animals.animal_0.eyes.eye_0.resolution.1")
    # plot_data(data, "env.animals.animal_0.eyes.eye_0.coord.0")
    # plot_data(data, "env.animals.animal_0.eyes.eye_0.coord.1")
    # plot_data(data, "env.animals.animal_0.eyes.eye_0.fov.0")
    # plot_data(data, "env.animals.animal_0.eyes.eye_0.fov.1")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse Nevergrad results")

    parser.add_argument("folder", type=Path)

    args = parser.parse_args()

    main(args.folder)