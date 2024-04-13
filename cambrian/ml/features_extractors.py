from typing import Dict, List
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

# ==================
# Utils


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


# ==================
# Feature Extractors


class MjCambrianBaseFeaturesExtractor(BaseFeaturesExtractor):
    pass


class MjCambrianCombinedExtractor(MjCambrianBaseFeaturesExtractor):
    """Overwrite of the default feature extractor of Stable Baselines 3."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        output_dim: int,
        normalized_image: bool,
        activation: nn.Module,
        image_extractor: MjCambrianBaseFeaturesExtractor,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put
        # something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, torch.nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                subspace = maybe_transpose_space(subspace, key)
                extractors[key] = image_extractor(
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

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            observation = maybe_transpose_obs(observations[key])
            encoded_tensor_list.append(extractor(observation))

        features = torch.cat(encoded_tensor_list, dim=1)
        return features


class MjCambrianImageFeaturesExtractor(MjCambrianBaseFeaturesExtractor):
    """This is a feature extractor for images. Will implement an image queue for
    temporal features. Should be inherited by other classes."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module,
    ):
        super().__init__(observation_space, features_dim)

        self.queue_size = observation_space.shape[0]
        self.temporal_linear = torch.nn.Sequential(
            torch.nn.Linear(features_dim * self.queue_size, features_dim),
            activation(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.temporal_linear(observations)


class MjCambrianMLPExtractor(MjCambrianImageFeaturesExtractor):
    """MLP feature extractor for small images. Essentially NatureCNN but with MLPs."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module,
        architecture: List[int],
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        n_input_channels = 3  # rgb
        height = observation_space.shape[2]
        width = observation_space.shape[3]
        self.num_pixels = n_input_channels * height * width

        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(self.num_pixels, architecture[0]))
        layers.append(activation())
        for i in range(1, len(architecture)):
            layers.append(torch.nn.Linear(architecture[i - 1], architecture[i]))
            layers.append(activation())
        layers.append(torch.nn.Linear(architecture[-1], features_dim))
        layers.append(activation())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        observations = observations.reshape(-1, self.num_pixels)  # [B, C * H * W]
        encodings = self.mlp(observations)
        encodings = encodings.reshape(B, -1)

        return super().forward(encodings)


class MjCambrianTransformerExtractor(MjCambrianImageFeaturesExtractor):
    """Transformer feature extractor for small images."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        self.n_input_channels = 3  # rgb
        height = observation_space.shape[2]
        width = observation_space.shape[3]
        self.num_pixels = self.n_input_channels * height * width
        self.input_dim = height * width  # Treat each pixel as a sequence element

        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.input_dim, self.n_input_channels)
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_input_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation(),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.final_linear = nn.Linear(self.n_input_channels, features_dim)
        self.activation = activation()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        observations = observations.reshape(-1, self.input_dim, self.n_input_channels)

        observations += self.positional_encoding
        encodings = self.transformer_encoder(observations)
        encodings = encodings.mean(dim=1)  # Pooling over sequence dimension
        encodings = self.final_linear(encodings)
        encodings = self.activation(encodings)
        encodings = encodings.reshape(B, -1)

        return super().forward(encodings)


class MjCambrianViTExtractor(MjCambrianImageFeaturesExtractor):
    class PatchEmbedding(nn.Module):
        def __init__(self, in_channels, patch_size, dim):
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Conv2d(
                in_channels, dim, kernel_size=patch_size, stride=patch_size
            )

        def forward(self, x):
            x = self.proj(x)  # B, C, H, W
            x = x.flatten(2)  # B, C, HW
            x = x.transpose(1, 2)  # B, HW, C
            return x

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: nn.Module,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        patch_size: int,
        emb_dim: int,
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        self.n_input_channels = 3  # RGB
        height, width = observation_space.shape[2], observation_space.shape[3]
        self.num_patches = (height // patch_size) * (width // patch_size)

        self.patch_embedding = self.PatchEmbedding(
            self.n_input_channels, patch_size, emb_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, emb_dim)
        )
        self.dropout = nn.Dropout(0.1)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=str(activation()).lower(),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )

        self.final_linear = nn.Linear(emb_dim, features_dim)
        self.activation = activation()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        x = self.patch_embedding(observations)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embeddings
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = x[:, 0]  # Using the cls_token

        x = self.final_linear(x)
        x = self.activation(x)

        return super().forward(x)


class MjCambrianNatureCNNExtractor(MjCambrianImageFeaturesExtractor):
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
        activation: nn.Module,
    ):
        super().__init__(observation_space, features_dim, activation)
        # We assume CxHxW images (channels first)

        n_input_channels = observation_space.shape[1]
        width, height = observation_space.shape[2], observation_space.shape[3]

        # Dynamically calculate kernel sizes and strides
        kernel_sizes, strides = self.calculate_dynamic_params(width, height)

        # Create CNN layers
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_input_channels, 32, kernel_size=kernel_sizes[0], stride=strides[0]
            ),
            activation(),
            torch.nn.Conv2d(32, 64, kernel_size=kernel_sizes[1], stride=strides[1]),
            activation(),
            torch.nn.Conv2d(64, 64, kernel_size=kernel_sizes[2], stride=strides[2]),
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

    def calculate_dynamic_params(self, width, height):
        # Define max sizes and strides based on your constraints
        max_kernel_sizes = [8, 4, 3]
        max_strides = [4, 2, 1]

        # Adjust kernel sizes and strides based on input dimensions
        kernel_sizes = [min(k, height, width) for k in max_kernel_sizes]
        strides = [
            min(s, height // k, width // k) for s, k in zip(max_strides, kernel_sizes)
        ]

        return kernel_sizes, strides

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        T = observations.shape[1]

        encoding_list = []
        for t in range(T):
            encoding = self.linear(self.cnn(observations[:, t]))
            encoding_list.append(encoding)

        return super().forward(torch.stack(encoding_list, dim=1))
