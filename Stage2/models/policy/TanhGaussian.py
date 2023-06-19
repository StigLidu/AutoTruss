from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

import abc
import logging
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class EmbedTanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            input_dims,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        self.embed_dim = input_dims[-1]
        # print(input_dims[-1])
        # print(input_dims[:-1])
        self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
        self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5)
        self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=2)
        self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=4, num_encoder_layers=2)

    def forward(self, obs):
        inputs = [obs, torch.ones(obs.shape[0], 3).cuda()]
        flat_inputs = torch.cat(inputs, dim=-1).cuda()
        # print(flat_inputs.shape)
        flat_dim = flat_inputs.shape[1]
        act_inputs = flat_inputs[..., -3:]
        id_inputs = flat_inputs[..., -5: -3]
        num_points = int((math.sqrt(25 + 8 * (flat_dim - 5)) - 5) / 2)
        # print(num_points)
        pos_inputs = flat_inputs[..., :2 * num_points]
        force_inputs = flat_inputs[..., -5 - num_points: -5]
        edge_inputs = flat_inputs[..., 2 * num_points: -5 - num_points]
        node_outputs = NodeEmbedding(pos_inputs, force_inputs, num_points).cuda()
        embed_node_outputs = self.embed_node(node_outputs)
        # print(node_outputs.shape, embed_node_outputs.shape)
        edge_outputs = EdgeEmbedding(pos_inputs, edge_inputs, num_points).cuda()
        embed_edge_outputs = self.embed_edge(edge_outputs)
        # print(edge_outputs.shape, embed_edge_outputs.shape)
        embed_id_outputs = self.embed_id(id_inputs).unsqueeze(1)
        embed_act_outputs = self.embed_act(act_inputs).unsqueeze(1)
        # print(embed_id_outputs.shape, embed_act_outputs.shape)

        src = torch.cat([embed_node_outputs, embed_edge_outputs, embed_id_outputs], dim=1).transpose(0, 1)
        # print(src.shape)
        tgt = embed_act_outputs.transpose(0, 1)
        # print(tgt.shape)
        outs = self.transformer(src, tgt).transpose(0, 1).squeeze(dim=1)
        #print(outs.shape)

        h = outs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


def EdgeEmbedding(pos_inputs, edge_inputs, num_points):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, 2)
    outputs = []
    for k in range(_pos_inputs.shape[0]):
        one_output = []
        idx = 0
        i = 0
        j = 1
        while idx < num_points * (num_points - 1) / 2:
            one_edge = []
            v_i = [_pos_inputs[k][i][0], _pos_inputs[k][i][1]]
            v_j = [_pos_inputs[k][j][0], _pos_inputs[k][j][1]]
            area_ij = edge_inputs[k][idx]
            one_edge += v_i
            one_edge += v_j
            one_edge.append(area_ij)
            idx += 1
            i += 1
            if i == j:
                i = 0
                j += 1
            one_output.append(one_edge)
        outputs.append(one_output)
    return torch.Tensor(outputs)

def NodeEmbedding(pos_inputs, force_inputs, num_points):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, 2)
    _force_inputs = force_inputs.reshape(force_inputs.shape[0], num_points, 1)
    return torch.cat([_pos_inputs, _force_inputs], dim=-1)