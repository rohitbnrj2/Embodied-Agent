from pathlib import Path
from typing import Union
import torch

from stable_baselines3 import PPO


class MjCambrianPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, path: Union[str, Path], *args, **kwargs):
        super().save(path, *args, **kwargs)
        self.save_policy(path)

    def save_policy(self, path: Union[str, Path]):
        """Overwrite the save method. Instead of saving the entire state, we'll
        just save the policy weights."""

        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), Path(path) / "policy.pt")

    def load_policy(self, path: Union[str, Path]):
        """Overwrite the load method. Instead of loading the entire state, we'll just
        load the policy weights."""

        Path(path).mkdir(parents=True, exist_ok=True)
        policy_path = Path(path) / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"Could not find policy.pt file at {policy_path}.")

        # Loop through the loaded state_dict and remove any layers that don't match in 
        # shape with the current policy
        saved_state_dict = torch.load(policy_path)
        policy_state_dict = self.policy.state_dict()
        for saved_state_dict_key in list(saved_state_dict.keys()):
            if saved_state_dict_key not in policy_state_dict:
                print(
                    f"WARNING: Key '{saved_state_dict_key}' not found in policy "
                    "state_dict. This shouldn't happen."
                )
                del saved_state_dict[saved_state_dict_key]

            saved_state_dict_var = saved_state_dict[saved_state_dict_key]
            policy_state_dict_var = policy_state_dict[saved_state_dict_key]

            if saved_state_dict_var.shape != policy_state_dict_var.shape:
                print(f"WARNING: Shape mismatch for key '{saved_state_dict_key}'")
                del saved_state_dict[saved_state_dict_key]

        self.policy.load_state_dict(saved_state_dict, strict=False)
