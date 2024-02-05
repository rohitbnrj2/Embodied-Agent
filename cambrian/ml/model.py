from typing import List, Any, Dict
from pathlib import Path
import pickle

import torch
from stable_baselines3 import PPO


class MjCambrianModel(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rollout: List[Dict[str, Any]] = None

    def save_policy(self, path: Path | str):
        """Overwrite the save method. Instead of saving the entire state, we'll
        just save the policy weights."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path / "policy.pt")

    def load_policy(self, path: Path | str):
        """Overwrite the load method. Instead of loading the entire state, we'll just
        load the policy weights.
        
        There are four cases to consider:
            - A layer in the saved policy is identical in shape to the current policy
                - Do nothing for this layer
            - A layer is both present in the saved policy and the current policy, but
                the shapes are different
                - Delete the layer from the saved policy
            - A layer is present in the saved policy but not the current policy
                - Delete the layer from the saved policy
            - A layer is present in the current policy but not the saved policy
                - Do nothing for this layer. By setting `strict=False` in the call to
                    `load_state_dict`, we can ignore this layer.
        """

        policy_path = Path(path) / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"Could not find policy.pt file at {policy_path}.")

        # Loop through the loaded state_dict and remove any layers that don't match in 
        # shape with the current policy
        saved_state_dict = torch.load(policy_path)
        policy_state_dict = self.policy.state_dict()
        for saved_state_dict_key in list(saved_state_dict.keys()):
            if saved_state_dict_key not in policy_state_dict:
                self.logger.warning(
                    f"Key '{saved_state_dict_key}' not found in policy "
                    "state_dict. Deleting from saved state dict." 
                )
                del saved_state_dict[saved_state_dict_key]
                continue

            saved_state_dict_var = saved_state_dict[saved_state_dict_key]
            policy_state_dict_var = policy_state_dict[saved_state_dict_key]

            if saved_state_dict_var.shape != policy_state_dict_var.shape:
                self.logger.warning(f"Shape mismatch for key '{saved_state_dict_key}'")
                del saved_state_dict[saved_state_dict_key]

        self.policy.load_state_dict(saved_state_dict, strict=False)

    def load_rollout(self, path: Path | str):
        """Load the rollout data from a previous training run. The rollout is a list
        of actions based on a current step. The model.predict call will then be 
        overwritten to return the next action. This loader is "dumb" in the sense that
        it doesn't actually process the observations when it's using rollout, it will 
        simply keep track of the current step and return the next action in the
        rollout.
        """
        with open(path, "rb") as f:
            self._rollout = pickle.load(f)["actions"]

    def predict(self, *args, **kwargs):
        if self._rollout is not None:
            return self._rollout.pop(0), None

        return super().predict(*args, **kwargs)