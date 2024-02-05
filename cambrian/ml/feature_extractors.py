from typing import Dict
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


def is_image_space(
    observation_space: gym.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """This is an extension of the sb3 is_image_space to support both regular images
    (HxWxC) and images with an additional dimension (NxHxWxC)."""
    from stable_baselines3.common.preprocessing import (
        is_image_space as sb3_is_image_space,
    )

    return len(observation_space.shape) == 4 or sb3_is_image_space(
        observation_space, normalized_image=normalized_image
    )


def maybe_transpose_space(observation_space: spaces.Box, key: str = "") -> spaces.Box:
    """This is an extension of the sb3 maybe_transpose_space to support both regular
    images (HxWxC) and images with an additional dimension (NxHxWxC). sb3 will call
    maybe_transpose_space on the 3D case, but not the 4D."""

    if len(observation_space.shape) == 4:
        num, height, width, channels = observation_space.shape
        new_shape = (num, channels, height, width)
        observation_space = spaces.Box(
            low=observation_space.low.reshape(new_shape),
            high=observation_space.high.reshape(new_shape),
            dtype=observation_space.dtype,
        )
    return observation_space


def maybe_transpose_obs(observation: torch.Tensor) -> torch.Tensor:
    """This is an extension of the sb3 maybe_transpose_obs to support both regular
    images (HxWxC) and images with an additional dimension (NxHxWxC). sb3 will call
    maybe_transpose_obs on the 3D case, but not the 4D.

    NOTE: in this case, there is a batch dimension, so the observation is 5D.
    """

    if len(observation.shape) == 5:
        observation = observation.permute(0, 1, 4, 2, 3)

    return observation


class MjCambrianCombinedExtractor(BaseFeaturesExtractor):
    """Overwrite of the default feature extractor of Stable Baselines 3."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        output_dim: int = 256,
        normalized_image: bool = True,
        activation: nn.Module | str = nn.Tanh,
        use_static_output: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put
        # something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        if isinstance(activation, str):
            assert hasattr(nn, activation), f"{activation} is not an activation"
            activation = getattr(nn, activation)

        extractors: Dict[str, torch.nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                subspace = maybe_transpose_space(subspace, key)
                extractors[key] = MjCambrianLowLevel(
                    subspace,
                    features_dim=output_dim,
                    activation=activation,
                )
                total_concat_size += output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = torch.nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        self.use_static_output = use_static_output
        if use_static_output:
            self._features_processing = nn.Sequential(
                nn.Linear(total_concat_size, 256),
                activation(),
                nn.Linear(256, self._features_dim),
                activation(),
            )

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            observation = maybe_transpose_obs(observations[key])
            encoded_tensor_list.append(extractor(observation))

        features = torch.cat(encoded_tensor_list, dim=1)
        if self.use_static_output:
            features = self._features_processing(features)
        return features


class MjCambrianLowLevel(BaseFeaturesExtractor):
    """MLP feature extractor for small images. Essentially NatureCNN but with MLPs."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module,
    ) -> None:
        # We assume TxHxWxC images (channels last as input)
        super().__init__(observation_space, features_dim)

        time_steps = observation_space.shape[0]
        n_input_channels = observation_space.shape[1]
        height = observation_space.shape[2]
        width = observation_space.shape[3]
        self.num_pixels = n_input_channels * height * width

        self.mlp = torch.nn.Sequential(
            nn.Flatten(),
            torch.nn.Linear(self.num_pixels, 64),
            activation(),
            torch.nn.Linear(64, 128),
            activation(),
            torch.nn.Linear(128, features_dim),
            activation(),
        )

        self.temporal_linear = torch.nn.Sequential(
            torch.nn.Linear(features_dim * time_steps, features_dim),
            activation(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        observations = observations.reshape(-1, self.num_pixels)  # [B, C * T * H * W]
        encodings = self.mlp(observations)
        encodings = encodings.view(B, -1)  # [B, T * features_dim]

        return self.temporal_linear(encodings)


class MjCambrianNatureCNN(BaseFeaturesExtractor):
    """This class overrides the default CNN feature extractor of Stable Baselines 3.

    The default feature extractor doesn't support images smaller than 36x36 because of
    the kernel_size, stride, and padding parameters of the convolutional layers. This
    class just overrides this functionality _only_ when the observation space has an
    image smaller than 36x36. Otherwise, it just uses the default feature extractor
    logic.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        time_steps = observation_space.shape[0]
        n_input_channels = observation_space.shape[1]
        if min(observation_space.shape[1:]) > 36:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                activation(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                activation(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                activation(),
                torch.nn.Flatten(),
            )
        else:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(n_input_channels, 32, kernel_size=1),
                activation(),
                torch.nn.Conv2d(32, 64, kernel_size=1),
                activation(),
                torch.nn.Conv2d(64, 64, kernel_size=1),
                activation(),
                torch.nn.Flatten(),
            )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None][:, 0])
            ).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim), activation()
        )

        self.temporal_linear = torch.nn.Sequential(
            torch.nn.Linear(features_dim * time_steps, features_dim),
            activation(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        T = observations.shape[1]

        encoding_list = []
        for t in range(T):
            encoding = self.linear(self.cnn(observations[:, t]))
            encoding_list.append(encoding)

        encodings = torch.cat(encoding_list, dim=1)
        return self.temporal_linear(encodings)


class MjCambrianMLP(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        time_steps = observation_space.shape[0]
        n_input_channels = observation_space.shape[3]
        x = (
            torch.as_tensor(observation_space.sample()[None][:, 0, :, :, :])
            .float()
            .permute(0, 3, 1, 2)
        )
        x = x[0, :, :, :]
        x = x.flatten()
        per_channel_input = int((list(x.shape)[0]) / n_input_channels)
        del x
        # Compute shape by doing one forward pass
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_channels * per_channel_input, 32 * per_channel_input),
            activation(),
            nn.Linear(32 * per_channel_input, 64 * per_channel_input),
            activation(),
            nn.Linear(64 * per_channel_input, 64 * per_channel_input),
            activation(),
        )
        with torch.no_grad():
            # print(f"{observation_space.sample()[None][:, 0, :, :, :].shape}")
            n_flatten = self.mlp(
                torch.as_tensor(observation_space.sample()[None][:, 0, :, :, :])
                .float()
                .permute(0, 3, 1, 2)
            ).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim), activation()
        )

        self.temporal_linear = torch.nn.Sequential(
            torch.nn.Linear(features_dim * time_steps, features_dim), activation()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        T = observations.shape[1]
        encoding_list = []  # [B, self.features_dim, T]
        for t in range(T):
            obs_ = observations[:, t, :, :, :].permute(0, 3, 1, 2)
            encoding = self.linear(self.mlp(obs_))
            encoding_list.append(encoding)
        # output should be of shape [B, self.features_dim]
        return self.temporal_linear(
            torch.cat(encoding_list, dim=1).reshape(B, -1)
        ).reshape(B, -1)
