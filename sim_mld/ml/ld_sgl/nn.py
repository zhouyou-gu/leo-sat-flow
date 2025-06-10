import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm

class PriceGNN(torch.nn.Module):
    def __init__(self, in_node_dim=3, in_edge_dim=1, hidden=64, num_layers=3, elu_offset=-1):
        super().__init__()
        # Encoders
        self.node_encoder = torch.nn.Linear(in_node_dim, hidden)
        self.edge_encoder = torch.nn.Linear(in_edge_dim, hidden)
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden*hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden*hidden, hidden*hidden),
            )
            self.convs.append(NNConv(hidden, hidden, nn_edge))
            self.bns.append(BatchNorm(hidden))
        # Decoders
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(3*hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.elu_offset = elu_offset
    def forward(self, x, edge_index, edge_attr):
        # Encode inputs
        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr.unsqueeze(-1))
        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, edge_index, e))
            x = bn(x)
        # Edge weight prediction
        row, col = edge_index
        z_edge = torch.cat([x[row], x[col], e], dim=-1)
        w = self.edge_decoder(z_edge).squeeze(-1)
        return F.elu(w + self.elu_offset) + 1
    
    
    