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
from rlkit.torch.sac.policies import TanhGaussianPolicy

class MLP(ConcatMlp):
    def __init__(self, input_dim, hidden_dims):
        super().__init__(input_size=input_dim, output_size=1, hidden_sizes=hidden_dims)

class _Mlp(Mlp):
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

class Transformer_Value_Network(nn.Module):
    """
        value network for UCTs
        hidden dims: hidden_dims for Mlp
        input dims: [... -1]: ... for Mlp for input size, -1 for enbedding size 
    """
    def __init__(self, input_dims, hidden_dims, env_dims, env_mode, num_node):
        super(Transformer_Value_Network, self).__init__()
        self.embed_dim = input_dims[-1]
        self.env_dims = env_dims
        self.env_mode = env_mode
        self.num_node = num_node
        self.num_edge = self.num_node * (self.num_node - 1) // 2
        # print(self.num_node * (self.env_dims + 1) + self.num_edge * (1 + 2 * self.env_dims))
        self.query_valid_embed = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 1)
        self.query_value_embed = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 1)
        if (env_dims == 2):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 3) #2 pos + 1 force = 3
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 5) #2 * 2 pos + 1 Area = 5
        if (env_dims == 3 and env_mode == 'Area'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 7) #2 * 3 pos + 1 Area = 7
        if (env_dims == 3 and env_mode == 'DT'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size = 8) #2 * 3 pos + 2 dt = 8
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead = 1, num_encoder_layers = 6)
        self.valid_head = Mlp(input_size = input_dims[-1], output_size = 2, hidden_sizes = hidden_dims)
        self.value_head = Mlp(input_size = input_dims[-1], output_size = 1, hidden_sizes = hidden_dims)

    def forward(self, *inputs, **kwargs): 
        flat_inputs = torch.cat(inputs, dim = -1).cuda()
        flat_dim = flat_inputs.shape[1]
        #print(flat_dim)
        #print(self.env_dims)
        #print(flat_inputs.shape)
        if (self.env_dims == 2):
            num_points = int((math.sqrt(25 + 8 * (flat_dim)) - 5) / 2) # check!!!
            pos_inputs = flat_inputs[..., :2 * num_points]
            force_inputs = flat_inputs[..., -num_points: ]
            edge_inputs = flat_inputs[..., 2 * num_points: -num_points]
        if (self.env_dims == 3 and self.env_mode == 'Area'):
            num_points = int((math.sqrt(49 + 8 * (flat_dim)) - 7) / 2) # check!!!
            #1/2 * (-7 + sqrt(49 + 8n))
            pos_inputs = flat_inputs[..., :3 * num_points]
            force_inputs = flat_inputs[..., num_points: ]
            edge_inputs = flat_inputs[..., 3 * num_points: -num_points]

        if (self.env_dims == 3 and self.env_mode == 'DT'):
            num_points = int((math.sqrt(9 + 4 * (flat_dim)) - 3) / 2) # check!!!
            #1 / 2 * (-3 + sqrt(9 + 4n))
            pos_inputs = flat_inputs[..., :3 * num_points]
            force_inputs = flat_inputs[..., -num_points: ]
            edge_inputs = flat_inputs[..., 3 * num_points: -num_points]

        #print(pos_inputs[0])
        #print(force_inputs[0])
        #print(edge_inputs[0])
        node_outputs = NodeEmbedding(pos_inputs, force_inputs, num_points).cuda()
        #print(node_outputs[0])
        embed_node_outputs = self.embed_node(node_outputs)
        #print(node_outputs.shape, embed_node_outputs.shape)
        edge_outputs, src_mask = EdgeEmbedding_with_Mask(pos_inputs, edge_inputs, num_points, self.env_dims, self.env_mode)
        #print(edge_outputs[0])
        embed_edge_outputs = self.embed_edge(edge_outputs)
        #print(edge_outputs.shape, embed_edge_outputs.shape)
        #print(embed_id_outputs.shape, embed_act_outputs.shape)
        src = torch.cat([embed_node_outputs, embed_edge_outputs], dim=1).transpose(0, 1)
        #print(src.shape)
        query_input = torch.ones(flat_inputs.shape[0]).unsqueeze(-1).unsqueeze(0).to(src.device)
        tgt = torch.cat((self.query_valid_embed(query_input), self.query_value_embed(query_input)), dim = 0)
        #print(src.shape, tgt.shape)
        #print(src_mask.shape)
        #print(src_mask)
        #print(src_mask[1])
        src_mask = ~src_mask
        outs = self.transformer(src, tgt, src_mask = src_mask)
        #print(outs.shape)
        valid = self.valid_head(outs[0])
        value = self.value_head(outs[1])
        return valid, value

class TRANSFORMEREMBED(_Mlp):
    """
        represented by edge sequences --> MLP embed to high dim --> transformer --> MLP to dim 1
    """
    def __init__(self, input_dims, hidden_dims, env_dims, env_mode, num_point = None, src_mask = False):
        super().__init__(input_size=input_dims[-1], output_size=1, hidden_sizes=hidden_dims)
        self.embed_dim = input_dims[-1]
        self.env_dims = env_dims
        self.env_mode = env_mode
        self.src_mask = src_mask
        if (env_dims == 2):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) #2 pos + 1 force = 3
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5) #2 * 2 pos + 1 Area = 5
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
            self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) # 2 pos + 1 Area = 3
        if (env_dims == 3 and env_mode == 'Area'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=7) #2 * 3 pos + 1 Area = 7
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) 
            self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) # 3 pos + 1 Area = 4  
        if (env_dims == 3 and env_mode == 'DT'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=8) #2 * 3 pos + 2 dt = 8
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) 
            self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5) # 3 pos + 2 dt = 5  
        if (src_mask):
            self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=1, num_encoder_layers=6)
        else: 
            self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=4, num_encoder_layers=2)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim).cuda()
        flat_dim = flat_inputs.shape[1]
        #print(flat_dim)
        #print(self.env_dims)
        #print(flat_inputs.shape)
        if (self.env_dims == 2):
            act_inputs = flat_inputs[..., -3:]
            id_inputs = flat_inputs[..., -6: -3]
            num_points = int((math.sqrt(25 + 8 * (flat_dim - 6)) - 5) / 2)
            pos_inputs = flat_inputs[..., :2 * num_points]
            force_inputs = flat_inputs[..., -6 - num_points: -6]
            edge_inputs = flat_inputs[..., 2 * num_points: -6 - num_points]
        if (self.env_dims == 3 and self.env_mode == 'Area'):
            act_inputs = flat_inputs[..., -4:]
            id_inputs = flat_inputs[..., -7: -4]
            num_points = int((math.sqrt(49 + 8 * (flat_dim - 7)) - 7) / 2)
            #1/2 * (-7 + sqrt(49 + 8n))
            pos_inputs = flat_inputs[..., :3 * num_points]
            force_inputs = flat_inputs[..., -7 - num_points: -7]
            edge_inputs = flat_inputs[..., 3 * num_points: -7 - num_points]
        if (self.env_dims == 3 and self.env_mode == 'DT'):
            act_inputs = flat_inputs[..., -5:]
            id_inputs = flat_inputs[..., -8: -5]
            num_points = int((math.sqrt(9 + 4 * (flat_dim - 8)) - 3) / 2)
            #1 / 2 * (-3 + sqrt(9 + 4n))
            pos_inputs = flat_inputs[..., :3 * num_points]
            force_inputs = flat_inputs[..., -8 - num_points: -8]
            edge_inputs = flat_inputs[..., 3 * num_points: -8 - num_points]

        node_outputs = NodeEmbedding(pos_inputs, force_inputs, num_points).cuda()
        embed_node_outputs = self.embed_node(node_outputs)
        #print(node_outputs.shape, embed_node_outputs.shape)
        edge_outputs, src_mask = EdgeEmbedding_with_Mask(pos_inputs, edge_inputs, num_points, self.env_dims, self.env_mode, with_act_id = True)
        edge_outputs.cuda() 
        src_mask.cuda()
        embed_edge_outputs = self.embed_edge(edge_outputs)
        #print(edge_outputs.shape, embed_edge_outputs.shape)
        embed_id_outputs = self.embed_id(id_inputs).unsqueeze(1)
        embed_act_outputs = self.embed_act(act_inputs).unsqueeze(1)
        #print(embed_id_outputs.shape, embed_act_outputs.shape)
        src = torch.cat([embed_node_outputs, embed_edge_outputs, embed_id_outputs], dim=1).transpose(0, 1)
        #print(src.shape)
        tgt = embed_act_outputs.transpose(0, 1)
        #print(tgt.shape)
        #print(src_mask.shape)
        #print(src_mask[0])
        if (not self.src_mask): src_mask = None
        if (src_mask != None): src_mask = ~src_mask
        outs = self.transformer(src, tgt, src_mask = src_mask).transpose(0, 1).squeeze(dim=1)
        #print(outs[0])
        return super().forward(outs, **kwargs)

def EdgeEmbedding(pos_inputs, edge_inputs, num_points, env_dims = 2, env_mode = 'Area'):# TODO:check it
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, -1)
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
            if (env_dims == 3):
                v_i = [_pos_inputs[k][i][0], _pos_inputs[k][i][1], _pos_inputs[k][i][2]]
                v_j = [_pos_inputs[k][j][0], _pos_inputs[k][j][1], _pos_inputs[k][j][2]]
            one_edge += v_i
            one_edge += v_j
            if (env_mode == 'Area'):
                area_ij = edge_inputs[k][idx]
                one_edge.append(area_ij)
            if (env_mode == 'DT'):
                d_ij = edge_inputs[k][idx * 2]
                t_ij = edge_inputs[k][idx * 2 + 1]
                one_edge.append(d_ij)
                one_edge.append(t_ij)
            idx += 1
            i += 1
            if i == j:
                i = 0
                j += 1
            one_output.append(one_edge)
        outputs.append(one_output)
    return torch.Tensor(outputs) # k * (num * (num - 1) / 2) * 8

def EdgeEmbedding_with_Mask(pos_inputs, edge_inputs, num_points, env_dims = 2, env_mode = 'Area', with_act_id = False):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, -1)
    outputs = []
    src_len = num_points + num_points * (num_points - 1) // 2
    if (not with_act_id):
        masks = torch.zeros((pos_inputs.shape[0], src_len, src_len), dtype = bool)
    else:
        masks = torch.zeros((pos_inputs.shape[0], src_len + 1, src_len + 1), dtype = bool)
    for k in range(_pos_inputs.shape[0]):
        if (not with_act_id):
            mask = torch.zeros((src_len, src_len), dtype = bool)
        else:
            mask = torch.zeros((src_len + 1, src_len + 1), dtype = bool)
        one_output = []
        idx = 0
        i, j = 0, 1
        while idx < num_points * (num_points - 1) // 2:
            one_edge = []
            v_i = [_pos_inputs[k][i][0], _pos_inputs[k][i][1]]
            v_j = [_pos_inputs[k][j][0], _pos_inputs[k][j][1]]
            if (env_dims == 3):
                v_i = [_pos_inputs[k][i][0], _pos_inputs[k][i][1], _pos_inputs[k][i][2]]
                v_j = [_pos_inputs[k][j][0], _pos_inputs[k][j][1], _pos_inputs[k][j][2]]
            one_edge += v_i
            one_edge += v_j
            if (env_mode == 'Area'):
                area_ij = edge_inputs[k][idx]
                one_edge.append(area_ij)
                if (area_ij > 0): 
                    mask[i, idx + num_points], mask[j, idx + num_points] = True, True
                    mask[idx + num_points, i], mask[idx + num_points, j] = True, True
            if (env_mode == 'DT'):
                d_ij = edge_inputs[k][idx * 2]
                t_ij = edge_inputs[k][idx * 2 + 1]
                one_edge.append(d_ij)
                one_edge.append(t_ij)
                if (d_ij > 0): 
                    mask[i, idx + num_points], mask[j, idx + num_points] = True, True
                    mask[idx + num_points, i], mask[idx + num_points, j] = True, True
            idx += 1
            i += 1
            if i == j:
                i = 0
                j += 1
            one_output.append(one_edge)
        outputs.append(one_output)
        masks[k] = mask
    if (with_act_id):
        for i in range(masks.shape[1]): masks[..., i, src_len], masks[..., src_len, i] = True, True
    for i in range(mask.shape[1]): masks[..., i, i] = True
    return torch.Tensor(outputs).cuda(), masks.cuda()

def NodeEmbedding(pos_inputs, force_inputs, num_points):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, -1) 
    _force_inputs = force_inputs.reshape(force_inputs.shape[0], num_points, -1)
    return torch.cat([_pos_inputs, _force_inputs], dim=-1)

# for flat_input, output is a sequence of tuple with size 5, each one is (P1.x, P1.y, P2.x, P2.y, area)
def Dim2EdgeEmbedding(flat_input, num_points): #UNUSED
    UseNormalizationEdge = False
    if UseNormalizationEdge:
        NormalizationEdge = 1000
    else:
        NormalizationEdge = 1
    output = []
    obs_action_dim = flat_input.size()[1]
    num_edges = int(num_points * (num_points - 1) / 2)
    obs_dim = num_points * 2 + num_edges
    act_dim = obs_action_dim - obs_dim
    fixed_points = int((obs_dim - act_dim) / 2)
    for one_input in flat_input:
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
            one_edge.append(one_input[idx] * NormalizationEdge)
            edges.append(one_edge)
            one_changed_edge = []
            one_changed_edge += changed_points[i]
            one_changed_edge += changed_points[j]
            one_changed_edge.append(one_input[changed_idx] * NormalizationEdge + one_input[idx] * NormalizationEdge)
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
    return torch.Tensor(output)

class Toy_Value_Network(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super(Toy_Value_Network, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.dropout = nn.Dropout(0.2)
        self.valid_head = nn.Linear(hidden_dims, 2)
        self.value_head = nn.Linear(hidden_dims, 1)

    def forward(self, *inputs, **kwargs): 
        flat_inputs = torch.cat(inputs, dim = -1).cuda()
        hidden = self.linear1(flat_inputs)
        hidden = self.ReLU(self.dropout(hidden))
        hidden = self.linear2(hidden)
        hidden = self.ReLU(self.dropout(hidden))
        valid = self.valid_head(hidden)
        value = self.value_head(hidden)
        return valid, value


class TRANSFORMEREMBED_policy(TanhGaussianPolicy):
    def __init__(self, input_dims, action_dim, hidden_dims, env_dims, env_mode, num_point = None, src_mask = False):
        super().__init__(obs_dim=input_dims[-1], action_dim = action_dim, hidden_sizes=hidden_dims)
        self.embed_dim = input_dims[-1]
        self.env_dims = env_dims
        self.env_mode = env_mode
        self.src_mask = src_mask
        self.num_point = num_point
        if (env_dims == 2):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) #2 pos + 1 force = 3
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5) #2 * 2 pos + 1 Area = 5
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
            #self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) # 2 pos + 1 Area = 3
        if (env_dims == 3 and env_mode == 'Area'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=7) #2 * 3 pos + 1 Area = 7
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) 
            #self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) # 3 pos + 1 Area = 4  
        if (env_dims == 3 and env_mode == 'DT'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=8) #2 * 3 pos + 2 dt = 8
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) 
            #self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5) # 3 pos + 2 dt = 5  
        if (src_mask):
            self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=1, num_encoder_layers=4)
        else: 
            self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=4, num_encoder_layers=2)

    def forward(self, obs):
        flat_inputs = obs
        #print(flat_dim)
        #print(self.env_dims)
        #print(flat_inputs.shape)
        assert(self.num_point != None)
        if (self.num_point != None): num_points = self.num_point
        if (self.env_dims == 2):
            id_inputs = flat_inputs[..., -3:]
            pos_inputs = flat_inputs[..., :2 * num_points]
            force_inputs = flat_inputs[..., -3 - num_points: -3]
            edge_inputs = flat_inputs[..., 2 * num_points: -3 - num_points]
        if (self.env_dims == 3 and self.env_mode == 'Area'):
            id_inputs = flat_inputs[..., -3:]
            pos_inputs = flat_inputs[..., :3 * num_points]
            force_inputs = flat_inputs[..., -3 - num_points: -3]
            edge_inputs = flat_inputs[..., 3 * num_points: -3 - num_points]
        if (self.env_dims == 3 and self.env_mode == 'DT'):
            id_inputs = flat_inputs[..., -3:]
            pos_inputs = flat_inputs[..., :3 * num_points]
            force_inputs = flat_inputs[..., -3 - num_points: -3]
            edge_inputs = flat_inputs[..., 3 * num_points: -3 - num_points]

        node_outputs = NodeEmbedding(pos_inputs, force_inputs, num_points).cuda()
        embed_node_outputs = self.embed_node(node_outputs)
        #print(node_outputs.shape, embed_node_outputs.shape)
        edge_outputs, src_mask = EdgeEmbedding_with_Mask(pos_inputs, edge_inputs, num_points, self.env_dims, self.env_mode, with_act_id = True)
        edge_outputs.cuda() 
        src_mask.cuda()
        embed_edge_outputs = self.embed_edge(edge_outputs)
        #print(edge_outputs.shape, embed_edge_outputs.shape)
        embed_id_outputs = self.embed_id(id_inputs).unsqueeze(1)
        #embed_act_outputs = self.embed_act(act_inputs).unsqueeze(1)
        #print(embed_id_outputs.shape, embed_act_outputs.shape)
        src = torch.cat([embed_node_outputs, embed_edge_outputs, embed_id_outputs], dim=1).transpose(0, 1)
        #print(src.shape)
        tgt = embed_id_outputs.transpose(0, 1)
        #print(tgt.shape)
        #print(src_mask.shape)
        #print(src_mask[0])
        if (not self.src_mask): src_mask = None
        if (src_mask != None): src_mask = ~src_mask
        outs = self.transformer(src, tgt, src_mask = src_mask).transpose(0, 1).squeeze(dim=1)
        #print(outs[0])
        return super().forward(outs)
    

if __name__ == '__main__':
    qf = TRANSFORMEREMBED([128, 256], [256, 512], 3, 'Area', 7).cuda()
    qf(torch.ones(1, 39).cuda())
