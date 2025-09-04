import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm, GATv2Conv

class PriceGNN(torch.nn.Module):
    def __init__(self, in_node_dim=2, in_edge_dim=1, hidden=64, num_layers=3, elu_offset=-1,
                    heads=4):
        super().__init__()
        # Encoders
        self.node_encoder = torch.nn.Linear(in_node_dim, hidden)
        # Keep an edge encoder for decoder context; GATv2 will take raw edge_attr with edge_dim=1
        self.edge_encoder = torch.nn.Linear(in_edge_dim, hidden)

        # GNN layers (GATv2 with edge features)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.edge_emb = torch.nn.ModuleList()
        for _ in range(num_layers):
            # concat=False keeps output dim == hidden regardless of heads
            self.convs.append(GATv2Conv(hidden, hidden, heads=heads, concat=False, edge_dim=hidden))
            self.bns.append(BatchNorm(hidden))
            self.edge_emb.append(torch.nn.Sequential(
            torch.nn.Linear(3 * hidden + in_node_dim * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden))
            )
        # Decoders
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden + in_node_dim * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.elu_offset = elu_offset

    def forward(self, x, edge_index, edge_attr):
        # Encode inputs
        xx = x
        x = self.node_encoder(x)
        # Edge embedding for decoder context
        e_dec = self.edge_encoder(edge_attr.unsqueeze(-1))

        for conv, bn, edge_emb in zip(self.convs, self.bns, self.edge_emb):
            x = F.elu(conv(x, edge_index, edge_attr=e_dec))
            x = bn(x)
            # row, col = edge_index
            # z_edge = torch.cat([
            #     x[row], xx[row],
            #     x[col], xx[col],
            #     e_dec
            # ], dim=-1)
            # e_dec = edge_emb(z_edge)

        # Edge weight prediction
        row, col = edge_index
        z_edge = torch.cat([
            x[row], xx[row],
            x[col], xx[col],
            e_dec
        ], dim=-1)
        w = self.edge_decoder(z_edge).squeeze(-1)
        return F.sigmoid(w)
    
    
    