from collections import OrderedDict
import torch
import torch.nn as nn
from gym import spaces
from typing import List, Dict, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class VariableMultiInputFeatureExtractor(BaseFeaturesExtractor):
    MULTIRES: int = 5 # what is this?
    EMBEDDER_OUTPUT_DIM: int = 64 # 

    def __init__(self, observation_space: spaces.Dict, features_dim: int, conv_dim: int):
        metadata_observation_space = {k: v for k, v in observation_space.items() if k != 'intensity'}
        super().__init__(observation_space, features_dim + self.EMBEDDER_OUTPUT_DIM * len(metadata_observation_space))

        self.metadata: Dict[str, Tuple[int, Embedder, nn.Sequential]] = {}

        for key, obs in metadata_observation_space.items():
            print(f"Adding {key} to MLP")
            embedder, dim = get_embedder(multires=self.MULTIRES, input_dim=obs.shape[1])
            dim *= obs.shape[0]

            self.metadata[key] = (
                dim,
                embedder, 
                nn.Sequential(OrderedDict([
                    (f"{key}_linear", nn.Linear(dim, self.EMBEDDER_OUTPUT_DIM)),
                    (f"{key}_relu", nn.ReLU()),
                ]))
            )


        # 3d 1x1x1 conv, avg pool, fc
        N,K,t = observation_space['intensity'].shape # eyes, photorecepetors, time buffer
        self.conv = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=(1,K), padding=(0, K // 2))
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.linear = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(int(N * int(K/2) * int(t/2)), 256)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(256, 128)),
            ('relu2', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(128, features_dim)),
        ]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B, N, K, t = observations['intensity'].shape

        ret = observations['intensity'].reshape(B,K,N,t)
        ret = self.conv(ret)
        ret = self.relu(ret)
        ret = self.pool(ret)
        ret = ret.reshape(B, -1)
        ret = self.linear(ret)

        for key, (dim, embedder, linear) in self.metadata.items():
            obs = observations[key]
            embed = embedder(obs).reshape(-1, dim)
            linear.to(obs.device) # why do I need this??????
            embed = linear(embed).reshape(B, -1)
            ret = torch.cat([ret, embed], dim=-1).reshape(B, -1)

        return ret.reshape(B, self.features_dim)

class MultiInputFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim):
        """
        Intensity: 
        Position: 
        Goal/Position
        """
        POS_INPUT_DIM = 2
        EYE_INPUT_DIM = 3
        MULTIRES = 5

        # features_dim = position + action + animal_config + features 
        self.f_dim = 0 
        _DIM_ = 64
        # import pdb; pdb.set_trace()

        if 'position_history' in observation_space.keys():
            print("Adding Position to MLP")
            self.pos_embeder, pos_out_dim = get_embedder(multires=MULTIRES, input_dim=POS_INPUT_DIM)
            self.f_dim += _DIM_
        else:
            self.pos_embeder = None
        
        if 'animal_config' in observation_space.keys(): 
            print("Adding Animal Config to MLP")
            self.eye_embeder, eye_out_dim = get_embedder(multires=MULTIRES, input_dim=EYE_INPUT_DIM)
            self.f_dim += _DIM_
        
        if 'action_history' in observation_space.keys():
            print("Adding Action to MLP")
            if self.pos_embeder == None: 
                self.pos_embeder, pos_out_dim = get_embedder(multires=MULTIRES, input_dim=POS_INPUT_DIM)
            self.f_dim += _DIM_

        self.f_dim += features_dim
        super().__init__(observation_space, self.f_dim)

        print("features_dim shape: {}".format(self.f_dim))

        # import pdb; pdb.set_trace()

        if 'position_history' in observation_space.keys():
            self._pos_dim = observation_space['position_history'].shape[0] * pos_out_dim # num_pixels x 22 
            self.pos_linear = nn.Sequential(nn.Linear(self._pos_dim , _DIM_), nn.ReLU())
        if 'animal_config' in observation_space.keys(): 
            self._eye_dim = observation_space['animal_config'].shape[0] * eye_out_dim # num_pixels x 33
            self.eye_linear = nn.Sequential(nn.Linear(self._eye_dim , _DIM_), nn.ReLU())
        if 'action_history' in observation_space.keys():
            self.action_linear = nn.Sequential(nn.Linear(self._pos_dim , _DIM_), nn.ReLU())

        # TODO: Currently we are processing everythign at once. we should take 
        # each eye's output and create a feature vector that is sent to a main layer for 
        # processing. This is scalable for multiple pixels. 
        self._obs_dim = observation_space['intensity'].shape[0] * observation_space['intensity'].shape[1] # obs_size x num_pixels
        self.linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self._obs_dim, 256)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(128, features_dim)),
            ]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B, N, C = observations['intensity'].shape
        obs_intensity = observations['intensity'].reshape(-1, self._obs_dim)
        ret = self.linear(obs_intensity)

        all_features = None

        if 'position_history' in observations:
            pos = observations['position_history']
            embed_pos = self.pos_embeder(pos).reshape(-1, self._pos_dim)
            feature_pos = self.pos_linear(embed_pos)
            all_features = feature_pos.reshape(B, -1)

        if 'action_history' in observations:
            act = observations['action_history']
            embed_action = self.pos_embeder(act).reshape(-1, self._pos_dim)
            feature_action = self.action_linear(embed_action)
            if all_features is None: 
                all_features = feature_action.reshape(B, -1)
            else:
                all_features = torch.cat([all_features, feature_action], dim=-1).reshape(B, -1)

        if 'animal_config' in observations: 
            an_cf = observations['animal_config']
            embed_eye = self.eye_embeder(an_cf).reshape(-1, self._eye_dim)
            feature_eye = self.eye_linear(embed_eye)
            if all_features is None: 
                all_features = feature_eye.reshape(B, -1)
            else:
                all_features = torch.cat([all_features, feature_eye], dim=-1).reshape(B, -1)

        ret = torch.cat([ret, all_features], dim=-1).reshape(B, self.f_dim)

        return ret

class CamSharedFeatureExtractor(BaseFeaturesExtractor):
    #weights of feature extractor are shared between cams
    def __init__(self, observation_space: spaces.Box, features_dim):
        #print("Observation Space Shape :", observation_space.shape)
        super().__init__(observation_space, features_dim)
        
        self.linear = nn.Sequential(nn.Linear(observation_space.shape[0] , 1), nn.ReLU())
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print("Observation Shape :", observations.shape)
        observations = torch.swapaxes(observations, 1, 2)
        ret = self.linear(observations)
        ret = torch.reshape(ret, (ret.shape[0], ret.shape[1]))
        return ret

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dim, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
