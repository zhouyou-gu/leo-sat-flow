import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.transformer_conv import *
from torch_geometric.nn import GCNConv, SimpleConv, MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops


class MessagePassingNNWithEdge(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1, hidden_channels=10):
        super(MessagePassingNNWithEdge, self).__init__(aggr='add')        
        # MLP that takes node features and edge attributes
        self.mlp = nn.Linear(2 * in_channels + edge_dim, out_channels,bias=False)
        self.update_mlp = nn.Linear(out_channels, out_channels,bias=False)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Concatenate target node, source node, and edge attributes
        combined = torch.cat([x_i, x_j, edge_attr], dim=1)

        messages = self.mlp(combined)
        return messages
    
    def update(self, aggr_out):
        updated = self.update_mlp(aggr_out)
        return updated


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, edge_dim, num_layers, output_dim=1, hidden_dim=10, activation=nn.ReLU(), conv="GCNConv", out_activation=nn.GELU()):
        super(GraphConstructor, self).__init__()
        
        self.edge_weight_lin = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim, bias=True),
            activation,
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            activation
        )

        self.i_lin = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(activation)
            tmp = nn.ModuleList()
            for _ in range(hidden_dim):
                if conv == "GCNConv":
                    tmp.append(GCNConv(1, 1, add_self_loops=False, normalize=True))
                elif conv == "TransformerConv":
                    tmp.append(TransformerConv(in_channels=1,out_channels=1,edge_dim=1))
                elif conv == "MessagePassingNNWithEdge":
                    tmp.append(MessagePassingNNWithEdge(in_channels=1,out_channels=1,edge_dim=1))
                elif conv == "SimpleConv":
                    tmp.append(SimpleConv())
                else:
                    raise Exception("Undefined Conv in GCNEvaluator")
            self.convs.append(tmp)
            self.convs.append(nn.Linear(hidden_dim+hidden_dim, hidden_dim, bias=True))
        
        self.edge_out_lin = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*4, bias=True),
            activation,
            nn.Linear(hidden_dim*4, hidden_dim*4, bias=True),
            activation,
            nn.Linear(hidden_dim*4, 1, bias=True),
            out_activation
        )
                
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for GCNEvaluator.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, input_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_dim].

        Returns:
            Tensor: Output tensor for each node of shape [num_nodes, output_dim].
        """
        edge_weight = self.edge_weight_lin(edge_attr).squeeze(-1)  # <-- Transform edge_attr to scalar weights

        x_ = self.i_lin(x)
        x = x_
        for layer in self.convs:
            if isinstance(layer, nn.ModuleList):
                x = torch.cat([e.forward(x[:,i].unsqueeze(-1), edge_index, edge_weight[:,i].unsqueeze(-1)) for i, e in enumerate(layer)], dim=1)
            elif isinstance(layer, nn.Linear):
                x = layer(torch.cat([x_,x],dim=1))
                x = x+x_
            else:
                x = layer(x)
        x = torch.cat([x_,x],dim=1)
        
        # 2. Gather embeddings for each edge
        src, dst = edge_index                 # edge_index is a [2, num_edges] tensor :contentReference[oaicite:6]{index=6}
        h_edge = torch.cat([x[src], x[dst]], dim=-1)  # [num_edges, 2*hidden_channels]
        
        # 3. Edge‐level prediction
        edge_out = self.edge_out_lin(h_edge)           # [num_edges, out_channels]
        
        return edge_out


import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d, Sequential, Linear, ReLU
from torch_geometric.nn import GINEConv, TopKPooling, global_mean_pool

class EdgeAwareConnectivityBackbone(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels, num_layers, pool_ratio=0.8):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        
        # Define an MLP for GINEConv that processes node+edge features
        gin_nn = Sequential(Linear(in_channels, hidden_channels), ReLU(),
                            Linear(hidden_channels, hidden_channels))
        self.convs.append(GINEConv(nn=gin_nn, edge_dim=edge_attr_dim ,train_eps=True))
        self.norms.append(BatchNorm1d(hidden_channels))
        
        # Additional layers
        for _ in range(num_layers - 1):
            gin_nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(),
                                Linear(hidden_channels, hidden_channels))
            self.convs.append(GINEConv(nn=gin_nn, train_eps=True))
            self.norms.append(BatchNorm1d(hidden_channels))
        
        # Pooling and readout
        self.pool = TopKPooling(hidden_channels, ratio=pool_ratio)
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            # Pass edge_attr to convolution
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = x + x_res  # residual connection
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch=batch)
        x = global_mean_pool(x, batch)
        return self.lin(x)


import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm

class PriceGNN(torch.nn.Module):
    def __init__(self, in_node_dim=2, in_edge_dim=1, hidden=64, num_layers=3):
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
        return F.elu(w-1)+1

class QrateGNN(torch.nn.Module):
    def __init__(self, in_node_dim=2, in_edge_dim=1, hidden=64, num_layers=3):
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
            torch.nn.Linear(2*hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
    def forward(self, x, edge_index, edge_attr, st_edge_index):
        # Encode inputs
        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr.unsqueeze(-1))
        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, edge_index, e))
            x = bn(x)
        # s-t edge weight prediction
        row, col = st_edge_index
        z_edge = torch.cat([x[row], x[col]], dim=-1)
        w = self.edge_decoder(z_edge).squeeze(-1)
        return F.elu(w-1)+1
    
    
    
class LpdGNN(torch.nn.Module):
    def __init__(self, in_node_dim=2, in_edge_dim=1, hidden=64, num_layers=3):
        super().__init__()
        self.price_gnn = PriceGNN(in_node_dim=in_node_dim, in_edge_dim=in_edge_dim, hidden=hidden, num_layers=num_layers)
        self.qrate_gnn = QrateGNN(in_node_dim=in_node_dim, in_edge_dim=in_edge_dim, hidden=hidden, num_layers=num_layers)

    def forward(self, x, edge_index, edge_attr, st_edge_index):
        qrates = self.qrate_gnn(x, edge_index, edge_attr, st_edge_index)
        prices = self.price_gnn(x, edge_index, edge_attr)
        return qrates, prices
