import torch, os, math
import numpy as np
from torch_geometric.nn import GCNConv
from torch import nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm, ConcatMlp, Mlp
from rlkit.torch.pytorch_util import activation_from_string
from torch_geometric.nn import CGConv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from torch_geometric.data import Data

class MLP(ConcatMlp):
    def __init__(self, input_dim, hidden_dims):
        super().__init__(input_size=input_dim, output_size=1, hidden_sizes=hidden_dims)

class _Mlp(Mlp):
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

class id_to_point:
    def __init__(self, num_point):
        self.trans = []
        u, v = 0, 1
        while (v < num_point):
            self.trans.append([u, v])
            v += 1
            if (v == num_point): u, v = u + 1, u + 2
    def convert(self, id):
        return self.trans[id]
    
class GNNEMBED(_Mlp):
    r"""
        represented by graph --> GCN --> MLP to dim 1
    """
    def __init__(self, input_dims, hidden_dims, env_dims, env_mode, num_point):
        super().__init__(input_size=input_dims[-1], output_size=1, hidden_sizes=hidden_dims)
        self.trans = id_to_point(num_point)
        self.embed_dim = input_dims[-1]
        self.env_dims = env_dims
        self.env_mode = env_mode
        #print(input_dims[-1])
        #print(input_dims[:-1])
        if (env_dims == 2):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) #2 pos + 1 force = 3
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5) #2 * 2 pos + 1 Area = 5
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
            self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) # 2 pos + 1 Area = 3
        if (env_dims == 3 and env_mode == 'Area'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=7) #2 * 3 pos + 1 Area = 7
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) # No change!!!
            self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) # 3 pos + 1 Area = 4     
        if (env_dims == 3 and env_mode == 'DT'):
            self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=4) #3 pos + 1 force = 4
            self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=8) #2 * 3 pos + 2 dt = 8
            self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3) # No change!!!
            self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5) # 3 pos + 2 dt = 5     
        if (env_mode == 'DT'):
            dim = 2
        else: dim = 1
        self.gcn1 = CGConv(channels=self.embed_dim, dim = dim)
        self.gcn2 = CGConv(channels=self.embed_dim, dim = dim)
        self.gcn3 = CGConv(channels=self.embed_dim, dim = dim)
        self.point_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.edge_query = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.greedy_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.mix_graph_action = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim).cuda()
        flat_dim = flat_inputs.shape[1]

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
        edge_id, edge_feature = Edge_IndexEmbedding(pos_inputs, edge_inputs, num_points, self.env_dims, self.env_mode)
        embed_graph_outs = torch.zeros(flat_inputs.shape[0], self.embed_dim).cuda()
        for i in range(flat_inputs.shape[0]):
            graph_data = Data(x = embed_node_outputs[i], edge_index = edge_id[i].transpose(0, 1), edge_attr = edge_feature[i])
            graph_out = self.gcn1(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            graph_data = Data(x = graph_out, edge_index = edge_id[i].transpose(0, 1), edge_attr = edge_feature[i])
            graph_out = self.gcn2(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            graph_data = Data(x = graph_out, edge_index = edge_id[i].transpose(0, 1), edge_attr = edge_feature[i])
            graph_out = self.gcn3(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            #print(edge_outputs.shape, embed_edge_outputs.shape)
            if (id_inputs[i, 0] != -1):
                embed_graph_out = self.point_query(graph_out[int(id_inputs[i, 0])])
            if (id_inputs[i, 1] != -1):
                u, v = self.trans.convert(int(id_inputs[i, 1]))
                embed_graph_out = self.edge_query(torch.cat((graph_out[u], graph_out[v])))
            if (id_inputs[i, 2] != -1):
                embed_graph_out = self.greedy_query(torch.max(graph_out, dim = -2)[0])
            embed_graph_outs[i] = embed_graph_out
        embed_act_outputs = self.embed_act(act_inputs)
        feature = self.mix_graph_action(torch.cat((embed_graph_outs, embed_act_outputs), dim = -1))
        x = super().forward(feature, **kwargs)
        return x
            
class GNNEMBED_policy(TanhGaussianPolicy):
    r"""
        represented by graph --> GCN --> MLP to dim 1
    """
    def __init__(self, input_dims, action_dim, hidden_dims, env_dims, env_mode, num_point):
        super().__init__(obs_dim=input_dims[-1], action_dim = action_dim, hidden_sizes=hidden_dims)
        self.trans = id_to_point(num_point)
        self.embed_dim = input_dims[-1]
        self.env_dims = env_dims
        self.env_mode = env_mode
        self.num_point = num_point
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
        if (env_mode == 'DT'):
            dim = 2
        else: dim = 1
        self.gcn1 = CGConv(channels=self.embed_dim, dim = dim)
        self.gcn2 = CGConv(channels=self.embed_dim, dim = dim)
        self.gcn3 = CGConv(channels=self.embed_dim, dim = dim)
        self.point_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.edge_query = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.greedy_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.mix_graph_action = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, obs):
        flat_inputs = obs
        num_points  = self.num_point
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
        edge_id, edge_feature = Edge_IndexEmbedding(pos_inputs, edge_inputs, num_points, self.env_dims, self.env_mode)
        embed_graph_outs = torch.zeros(flat_inputs.shape[0], self.embed_dim).cuda()
        for i in range(flat_inputs.shape[0]):
            graph_data = Data(x = embed_node_outputs[i], edge_index = edge_id[i].transpose(0, 1), edge_attr = edge_feature[i])
            graph_out = self.gcn1(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            graph_data = Data(x = graph_out, edge_index = edge_id[i].transpose(0, 1), edge_attr = edge_feature[i])
            graph_out = self.gcn2(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            graph_data = Data(x = graph_out, edge_index = edge_id[i].transpose(0, 1), edge_attr = edge_feature[i])
            graph_out = self.gcn3(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            #print(edge_outputs.shape, embed_edge_outputs.shape)
            if (id_inputs[i, 0] != -1):
                embed_graph_out = self.point_query(graph_out[int(id_inputs[i, 0])])
            if (id_inputs[i, 1] != -1):
                u, v = self.trans.convert(int(id_inputs[i, 1]))
                embed_graph_out = self.edge_query(torch.cat((graph_out[u], graph_out[v])))
            if (id_inputs[i, 2] != -1):
                embed_graph_out = self.greedy_query(torch.max(graph_out, dim = -2)[0])
            embed_graph_outs[i] = embed_graph_out
        feature = self.mix_graph_action(embed_graph_outs)
        x = super().forward(feature)
        return x

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
            j += 1
            if j == num_points:
                i += 1
                j = i + 1
            one_output.append(one_edge)
        outputs.append(one_output)
    return torch.Tensor(outputs) # k * (num * (num - 1) / 2) * 8

def Edge_IndexEmbedding(pos_inputs, edge_inputs, num_points, env_dims = 2, env_mode = 'Area'):# TODO:check it
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, -1)
    outputs = []
    feature_outputs = []
    for k in range(_pos_inputs.shape[0]):
        one_output = []
        one_feature_output = []
        idx = 0
        i = 0
        j = 1
        while idx < num_points * (num_points - 1) / 2:
            if (env_mode == 'Area' and edge_inputs[k][idx] > 0): 
                one_output.append([i, j])
                one_feature_output.append([edge_inputs[k][idx]])
            if (env_mode == 'DT' and edge_inputs[k][idx * 2] > 0): 
                one_output.append([i, j])
                one_feature_output.append([edge_inputs[k][idx * 2], edge_inputs[k][idx * 2 + 1]])
            idx += 1
            j += 1
            if j == num_points:
                i += 1
                j = i + 1
        outputs.append(torch.tensor(one_output).cuda())
        feature_outputs.append(torch.tensor(one_feature_output).cuda())
    return outputs, feature_outputs

def EdgeEmbedding_with_Mask(pos_inputs, edge_inputs, num_points, env_dims = 2, env_mode = 'Area'):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, -1)
    outputs = []
    src_len = num_points + num_points * (num_points - 1) // 2
    masks = torch.zeros((pos_inputs.shape[0], src_len, src_len), dtype = bool)
    for k in range(_pos_inputs.shape[0]):
        mask = torch.zeros((src_len, src_len), dtype = bool)
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
    return torch.Tensor(outputs).cuda(), masks.cuda()

def NodeEmbedding(pos_inputs, force_inputs, num_points): #TODO: check it 
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
        #output.append(together_edges)
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

if __name__ == '__main__':
    qf = GNNEMBED([128, 256], [256, 512], 3, 'Area', 7).cuda()
    qf(torch.ones(1, 39).cuda())
