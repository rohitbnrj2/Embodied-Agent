import argparse
from pathlib import Path

import torch

from cambrian.ml.model import MjCambrianModel
from cambrian.utils.wrappers import make_wrapped_env
from cambrian.utils.config import MjCambrianConfig

import torchviz_new as torchviz


def main(model_path: Path, config_file: Path | None = None):
    assert model_path.exists(), f"Model not found at {model_path}"
    if config_file is None:
        config_file = model_path.parent / "config.yaml"
    assert config_file.exists(), f"Config file not found at {config_file}"

    # Load the configuration
    config = MjCambrianConfig.load(config_file, instantiate=True)

    # Create the environment
    wrappers = [w for w in config.trainer.wrappers.values() if w]
    wrapped_env = make_wrapped_env(
        config=config.env.copy(),
        name=config.expname,
        wrappers=wrappers,
    )()

    # Load the model
    model = MjCambrianModel.load(model_path, env=wrapped_env)
    policy = model.policy
    policy.eval()

    # Create a sample input tensor with the appropriate observation space shape
    sample_input = model.observation_space.sample()
    for key, value in sample_input.items():
        sample_input[key] = (
            torch.from_numpy(value).unsqueeze(0).to(device=policy.device)
        )

    # Forward the sample input through the policy network to generate a graph
    output = policy(sample_input)
    dot = torchviz.make_dot(
        output, params=dict(policy.named_parameters()), show_saved=True
    )
    dot.format = "png"  # Specify the file format
    dot.render("model_visualization")  # Saves as "model_visualization.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a model using torchviz.")
    parser.add_argument("model_path", type=Path, help="Path to the saved model.")
    parser.add_argument(
        "config_file",
        type=Path,
        nargs="?",
        help="Path to the configuration file.",
        default=None,
    )
    args = parser.parse_args()

    main(args.model_path, args.config_file)
