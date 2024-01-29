from typing import Dict
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space


class MjCambrianCombinedExtractor(BaseFeaturesExtractor):
    """Overwrite of the default feature extractor of Stable Baselines 3."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        features_dim = 128
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put
        # something there. This is dirty!
        super().__init__(observation_space, features_dim = features_dim)

        extractors: Dict[str, torch.nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 4 or is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = MjCambrianMLP(
                    subspace,
                    features_dim=cnn_output_dim,
                    activation=nn.Tanh,
                    normalized_image=normalized_image,
                )
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = torch.nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = 256
        self._features_processing = nn.Sequential(nn.Linear(total_concat_size, 256), 
                                                  nn.Tanh(),
                                                  nn.Linear(256, self._features_dim), 
                                                  nn.Tanh()
                                                  )

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        initial_features = torch.cat(encoded_tensor_list, dim=1)
        processed_features = self._features_processing(initial_features)
        return processed_features


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
        features_dim: int = 512,
        activation = nn.ReLU,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # assert is_image_space(
        #     observation_space, check_channels=False, normalized_image=normalized_image
        # ), (
        #     "You should use NatureCNN "
        #     f"only with images not with {observation_space}\n"
        #     "(you are probably using `CnnPolicy` instead of `MlpPolicy` "
        #     "or `MultiInputPolicy`)\n"
        #     "If you are using a custom environment,\n"
        #     "please check it using our env checker:\n"
        #     "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
        #     "If you are using `VecNormalize` or already normalized "
        #     "channel-first images you should pass `normalize_images=False`: \n"
        #     "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        # )
        print("Using CNN!")
        time_steps = observation_space.shape[0]
        n_input_channels = observation_space.shape[3]
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
                torch.as_tensor(observation_space.sample()[None][:, 0, :, :, :]).float().permute(0, 3, 1, 2)
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
        encoding_list = [] # [B, self.features_dim, T]
        for t in range(T):
            obs_ = observations[:, t, :, :, :].permute(0, 3, 1, 2)
            encoding = self.linear(self.cnn(obs_))
            encoding_list.append(encoding)
        # output should be of shape [B, self.features_dim]
        return self.temporal_linear(torch.cat(encoding_list, dim=1).reshape(B, -1)).reshape(B, -1)

class MjCambrianMLP(BaseFeaturesExtractor):
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
        features_dim: int = 512,
        activation = nn.ReLU,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        time_steps = observation_space.shape[0]
        n_input_channels = observation_space.shape[3]
        x = torch.as_tensor(observation_space.sample()[None][:, 0, :, :, :]).float().permute(0, 3, 1, 2)
        x = x[0, : , : , :]
        x = x.flatten()
        per_channel_input = int((list(x.shape)[0])/n_input_channels)
        del x
        # Compute shape by doing one forward pass
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(n_input_channels*per_channel_input, 32*per_channel_input),
                                 activation(),
                                 nn.Linear(32*per_channel_input, 64*per_channel_input),
                                 activation(),
                                 nn.Linear(64*per_channel_input, 64*per_channel_input),
                                 activation())
        with torch.no_grad():
            #print(f"{observation_space.sample()[None][:, 0, :, :, :].shape}")
            n_flatten = self.mlp(
                torch.as_tensor(observation_space.sample()[None][:, 0, :, :, :]).float().permute(0, 3, 1, 2)
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
        encoding_list = [] # [B, self.features_dim, T]
        for t in range(T):
            obs_ = observations[:, t, :, :, :].permute(0, 3, 1, 2) / 255.
            encoding = self.linear(self.mlp(obs_))
            encoding_list.append(encoding)
        # output should be of shape [B, self.features_dim]
        return self.temporal_linear(torch.cat(encoding_list, dim=1).reshape(B, -1)).reshape(B, -1)

