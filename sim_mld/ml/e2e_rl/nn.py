import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm

from sim_mld.ml.ld_sgl.nn import PriceGNN

class AC_NN(torch.nn.Module):
    def __init__(self, in_node_dim=2, in_edge_dim=1, action_dim=1, hidden=64, num_layers=3):
        super().__init__()
        self.actor = PriceGNN(in_node_dim=in_node_dim, in_edge_dim=in_edge_dim, ot_edge_dim=action_dim, hidden=hidden, num_layers=num_layers)
        self.critic = PriceGNN(in_node_dim=in_node_dim, in_edge_dim=in_edge_dim+action_dim, ot_edge_dim=1, hidden=hidden, num_layers=num_layers)

    def forward(self, x, edge_index, edge_attr):
        action = self.actor(x, edge_index, edge_attr, dec_active_func=F.sigmoid)
        return action

    def evaluate(self, x, edge_index, edge_attr, action, st_pair_index):
        edge_attr = torch.cat([edge_attr.reshape(-1,1), action.reshape(-1,1)], dim=-1)
        q_value = self.critic(x, edge_index, edge_attr, dec_pair_index=st_pair_index, dec_active_func=F.elu)
        return q_value
