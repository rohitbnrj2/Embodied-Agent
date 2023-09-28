from typing import Union, Any, Dict
from pathlib import Path
from prodict import Prodict
import yaml

def read_yaml(filename: Union[Path, str]) -> Dict:
    return yaml.safe_load(Path(filename).read_text())

def write_yaml(config: Any, filename: Union[Path, str]):
    with open(filename, "w") as f:
        yaml.dump(config, f)

def load_config(config_file: Union[Path, str, Prodict]) -> Prodict:
    if isinstance(config_file, (Path, str)):
        config_file = Prodict.from_dict(read_yaml(config_file))
    return config_file
