import torch, os, math
import numpy as np
from torch import nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm, ConcatMlp, Mlp
from rlkit.torch.pytorch_util import activation_from_string

class MLP(ConcatMlp):
    def __init__(self, input_dim, hidden_dims):
        super().__init__(input_size=input_dim, output_size=1, hidden_sizes=hidden_dims)

class _Mlp(Mlp):
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

class TRANSFORMEREMBED(_Mlp):
    """
        represented by edge sequences --> MLP embed to high dim --> transformer --> MLP to dim 1
    """
    def __init__(self, input_dims, hidden_dims):
        super().__init__(input_size=input_dims[-1], output_size=1, hidden_sizes=hidden_dims)
        self.embed_dim = input_dims[-1]
        #print(input_dims[-1])
        #print(input_dims[:-1])
        self.embed = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5)
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=4, num_encoder_layers=2)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        obs_dim = inputs[0].shape[1]
        num_points = int((math.sqrt(9 + 8 * (obs_dim - 2)) -  3) / 2)
        outputs = Dim2EdgeEmbedding(flat_inputs, num_points).transpose(0, 1).cuda()
        #print('1', outputs.shape)
        outputs = self.embed(outputs)
        #print('2', outputs.shape)
        tgt = (torch.ones((outputs.shape[1], 1, self.embed_dim)).transpose(0, 1) / math.sqrt(self.embed_dim)).cuda()
        #print('3', tgt)
        outs = self.transformer(outputs, tgt).transpose(0, 1).squeeze(dim=1)
        #print('4', outs)
        #print('outs shape', outs.shape)
        return super().forward(outs, **kwargs)


# for flat_input, output is a sequence of tuple with size 5, each one is (P1.x, P1.y, P2.x, P2.y, area)
def Dim2EdgeEmbedding(flat_input, num_points):
    output = []
    obs_action_dim = flat_input.size()[1]
    num_edges = int(num_points * (num_points - 1) / 2)
    obs_dim = num_points * 2 + num_edges
    act_dim = obs_action_dim - obs_dim
    fixed_points = int((obs_dim - act_dim) / 2)
    for one_input in flat_input:
        for i in one_input:
            print(i)
        os._exit(0)
        points = []
        changed_points = []
        for i in range(num_points):
            points.append([one_input[2 * i], one_input[2 * i + 1]])
        for i in range(num_points):
            if i < fixed_points:
                changed_points.append(points[i])
            else:
                changed_points.append([points[i][0] + one_input[(i - fixed_points) * 2 + obs_dim], points[i][1] + one_input[(i - fixed_points) * 2 + obs_dim + 1]])

        together_edges = []
        edges = []
        changed_edges = []
        idx = 2 * num_points
        changed_idx = obs_dim + 2 * (num_points - fixed_points)
        i = 0
        j = 1
        while idx < obs_dim:
            one_edge = []
            one_edge += points[i]
            one_edge += points[j]
            one_edge.append(one_input[idx])
            edges.append(one_edge)
            one_changed_edge = []
            one_changed_edge += changed_points[i]
            one_changed_edge += changed_points[j]
            one_changed_edge.append(one_input[changed_idx] + one_input[idx])
            changed_edges.append(one_changed_edge)
            together_edges.append(one_edge)
            together_edges.append(one_changed_edge)
            idx += 1
            changed_idx += 1
            i += 1
            if i >= j:
                j += 1
                i = 0
        output.append(edges + [[-1, -1, -1, -1, -1],] + changed_edges)
        #output.append(together_edges)
    return torch.Tensor(output)
