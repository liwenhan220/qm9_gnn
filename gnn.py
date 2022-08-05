import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import MessagePassing
from mlp import MLP

class GNN_layer(MessagePassing):
    def __init__(self, input_channel, output_channel, edge_features):
        super().__init__(aggr='add', flow='source_to_target')
        self.phi_e = MLP(input_channel*2+edge_features, 1)
        self.phi_h = MLP(input_channel+1, output_channel)
        self.relu = nn.ReLU()

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        inputs = (torch.cat((x_i, x_j, edge_attr), dim=1))
        output = self.phi_e(inputs)
        return self.relu(output)

    def forward(self, x, edge_index, edge_attr):
        outputs = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        inputs = torch.cat((x, outputs), dim=1)
        return self.phi_h(inputs)

class NN(nn.Module):
    def __init__(self, node_features = 11, edge_features = 4, hidden_channels=128):
        super(NN, self).__init__()
        self.gnn1 = GNN_layer(node_features, hidden_channels, edge_features)
        self.gnn2 = GNN_layer(hidden_channels, hidden_channels, edge_features)
        self.gnn3 = GNN_layer(hidden_channels, hidden_channels, edge_features)
        self.gnn4 = GNN_layer(hidden_channels, hidden_channels, edge_features)
        self.gnn5 = GNN_layer(hidden_channels, hidden_channels, edge_features)
        self.gnn6 = GNN_layer(hidden_channels, hidden_channels, edge_features)
        self.gnn7 = GNN_layer(hidden_channels, hidden_channels, edge_features)
        self.silu = nn.ReLU()
        self.linear = nn.Linear(hidden_channels, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch  = data.x, data.edge_index, data.batch

        x = self.gnn1(x, edge_index, data.edge_attr)
        x = self.silu(x)

        x = self.gnn2(x, edge_index, data.edge_attr)
        x = self.silu(x)

        x = self.gnn3(x, edge_index, data.edge_attr)
        x = self.silu(x)

        # x = self.gnn4(x, edge_index, data.edge_attr)
        # x = self.silu(x)

        # x = self.gnn5(x, edge_index, data.edge_attr)
        # x = self.silu(x)

        # x = self.gnn6(x, edge_index, data.edge_attr)
        # x = self.silu(x)

        # x = self.gnn7(x, edge_index, data.edge_attr)
        # x = self.silu(x)

        x = global_add_pool(x, batch)

        x = self.linear(x)
        x = self.silu(x)

        x = self.linear2(x)
        return x


