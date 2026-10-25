"""
Asymmetric actor-critic for SB3 PPO: the critic sees the privileged true-state tail
of the observation, the actor sees it zeroed. Only the actor runs on the real car,
which fills that tail with zeros.
"""

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class MaskExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, policy_dim: int):
        super().__init__(observation_space, observation_space.shape[0])
        mask = torch.zeros(observation_space.shape[0])
        mask[:policy_dim] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, obs):
        return obs * self.mask


class PrivilegedCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, policy_dim: int, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)
        # same output width as the critic's flatten extractor, so one MlpExtractor serves both
        self.pi_features_extractor = MaskExtractor(self.observation_space, policy_dim)


class RecurrentPrivilegedCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, policy_dim: int, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)
        self.pi_features_extractor = MaskExtractor(self.observation_space, policy_dim)
