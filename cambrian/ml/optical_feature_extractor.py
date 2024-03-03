from typing import Dict

import numpy as np
from cambrian.ml.features_extractors import MjCambrianLowLevel
from cambrian.optics import MjCambrianNonDifferentiableOptics, add_gaussian_noise, electric_field, rs_prop
from cambrian.utils.config import MjCambrianAnimalConfig, MjCambrianEyeConfig
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Deque

from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space


class MjCambrianOpticalFeatureExtractor(BaseFeaturesExtractor):
    """Overwrite of the default feature extractor of Stable Baselines 3."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        config: MjCambrianAnimalConfig,
        output_dim: int = 256,
        normalized_image: bool = True,
        features_dim: int = 128,
        activation: nn.Module | str = nn.Tanh,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put
        # something there. This is dirty!
        super().__init__(observation_space, features_dim=features_dim)

        if isinstance(activation, str):
            assert hasattr(nn, activation), f"{activation} is not an activation"
            activation = getattr(nn, activation)

        extractors: Dict[str, torch.nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if "depth" in key:
                continue
            if len(subspace.shape) == 4 or is_image_space(
                subspace, normalized_image=normalized_image
            ):
                extractors[key] = MjCambrianLowLevel(
                    subspace,
                    features_dim=output_dim,
                    activation=activation,
                    normalized_image=normalized_image,
                )
                total_concat_size += output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = torch.nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        
        raise NotImplementedError("This is not implemented yet")

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            # check if observations is an image 
            if len(observations[key].shape) == 5:
                observation = self.forward_optics(key, observations[key].permute(0, 1, 4, 2, 3), observations[f"{key}_depth"])
                encoded_tensor_list.append(extractor(observation))
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

    def _parse_config(self, config: MjCambrianAnimalConfig):
        """Parse the configuration.
        """
        self.config = config
        self._eyes: Dict[str, Dict] = {}

        for i, eye_config in enumerate(self.config.eye_configs.values()):
            self._eyes[eye_config.name] = {}
            A, X1, Y1, FX, FY = self.define_simple_psf(eye_config.depth, eye_config)
            self._eyes[eye_config.name]["A"] = A
            self._eyes[eye_config.name]["X1"] = X1
            self._eyes[eye_config.name]["Y1"] = Y1
            self._eyes[eye_config.name]["FX"] = FX
            self._eyes[eye_config.name]["FY"] = FY
            self._eyes[eye_config.name]["focal"] = eye_config.focal
            self._eyes[eye_config.name]["wavelengths"] = eye_config.wavelengths

    def forward_optics(self, key: str, observation: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass through the optics.
        key: str, name of the eye
        observation: Assumes that the observation is of shape [batch, time, channels, height, width]
        depth: Assumes that the depth is of shape [batch, time, height, width]

        NOTE: this is actually much more expensive than the current approach because 
        it recalculates the PSF for every Time and the input is a moving image.
        """
        if self.config.add_noise:
            observation = add_gaussian_noise(observation, self.config.noise_std)

        B = observation.shape[0]
        T = observation.shape[1]
        observations = [] 
        for b in range(B):
            psf_b = []
            for t in range(T):
                z1 = depth[b].mean(dim=(-1,-2))
                psf = self.depth_invariant_psf(z1[b], 
                                               self._eyes[key]["A"], 
                                               self._eyes[key]["X1"], 
                                               self._eyes[key]["Y1"], 
                                               self._eyes[key]["FX"], 
                                               self._eyes[key]["FY"], 
                                               self._eyes[key]["focal"],
                                               self._eyes[key]["wavelengths"]
                                               )
                psf_b.append(psf.unsqueeze(0)) # [T, H, W, C]
            psf_b = torch.cat(psf_b, dim=0)
            _obs = torch.nn.functional.conv2d(observation[b], psf_b, padding='same')
            observations.append(_obs.unsqueeze(0)) # [B, T, H, W, C]
        
        observations = torch.cat(observations, dim=0)
        return observations

    def define_simple_psf(self, config: MjCambrianEyeConfig) -> torch.Tensor:
        """Define a simple point spread function (PSF) for the eye.
        """
                
        # Create Sensor Plane
        Mx = config.sensor_resolution[0] 
        My = config.sensor_resolution[1] 

        # id mx and my are even, then change to odd
        # odd length is better for performance
        if Mx % 2 == 0:
            Mx += 1
        if My % 2 == 0:
            My += 1
        
        Lx = config.sensorsize[0] # length of simulation plane (m) 
        Ly = config.sensorsize[1] # length of simulation plane (m) 
        dx = Lx/Mx # pixel pitch of sensor (m)      
        dy = Ly/My # pixel pitch of sensor (m)

        # Image plane coords                              
        x1 = np.linspace(-Lx/2.,Lx/2.,Mx) 
        y1 = np.linspace(-Ly/2.,Ly/2.,My) 
        X1,Y1 = np.meshgrid(x1,y1)

        # Frequency coords
        fx = np.linspace(-1./(2.*dx),1./(2.*dx),Mx)
        fy = np.linspace(-1./(2.*dy),1./(2.*dy),My)
        FX,FY = np.meshgrid(fx,fy)
        
        # Aperture
        max_aperture_size = dx * int(np.maximum(Mx, My) / 2) # (m)
        aperture_radius = np.interp(np.clip(config.aperture_open, 0, 1), [0, 1], [0, max_aperture_size])
        A = (np.sqrt(X1**2+Y1**2)/(aperture_radius + 1.0e-7) <= 1.).astype(np.float32)
        return A, X1, Y1, FX, FY

    def depth_invariant_psf(self, mean_depth, A, X1, Y1, FX, FY, focal, wavelengths) -> torch.Tensor:
        """
        mean_depth: float, mean depth of the point source
        """
        z1 = mean_depth # z1 is average distance of point source
        psfs = []
        for _lambda in wavelengths:
            k = 2*np.pi/_lambda
            # electric field originating from point source
            u = electric_field(k, z1, X1, Y1)
            # electric field at the aperture
            u = u * A #*t_lens*t_mask
            # electric field at the sensor plane
            u = rs_prop(u, focal[0], FX, FY, _lambda)
            psf = np.abs(u)**2
            # psf should sum to 1 because of energy 
            psf /= (np.sum(psf) + 1.0e-7) 
            psfs.append(torch.tensor(psf).unsqueeze(-1))
        return torch.cat(psfs, dim=-1).float()
