from typing import List
from dataclasses import field

from cambrian.utils.logger import get_logger
from cambrian.utils.config import (
    MjCambrianBaseConfig,
    MjCambrianConfig,
    run_hydra,
    config_wrapper,
)

@config_wrapper
class AnimalShowcaseConfig(MjCambrianBaseConfig):
    """The configuration for the animal showcase.
    
    Attributes:
        expermient (str): The experiment to run. This is the path to the hydra exp file 
            as if you are you running the experiment from the root of the project 
            (i.e. relative to the exp/ directory).

        overrides (List[str]): The overrides to apply to the loaded configuration.
    """

    experiment: str

    overrides: List[str]

def main(config: AnimalShowcaseConfig):
    print(config)
    pass

if __name__ == "__main__":
    run_hydra(main, config_name="tools/paper/animal_showcase")