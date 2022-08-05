import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import MessagePassing
from mlp import MLP

class GNN_layer(MessagePassing):
    def __init__(self, input_channel, output_channel, edge_features):
        super().__init__(aggr='add', flow='target_to_source')
        self.phi_e = MLP(input_channel*2+edge_features+1, 1)
        self.phi_h = MLP(input_channel+1, output_channel)
        self.phi_x = MLP(1, 1)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        xi_coords = x_i[:,x_i.shape[1]-3:]
        xj_coords = x_j[:,x_j.shape[1]-3:]
        
        diff = xi_coords - xj_coords
        sq = torch.pow(diff, 2.0)
        total = torch.sum(sq, dim=1)
        results = torch.pow(total, 0.5)
        results = results.reshape((len(results), 1))

        x_i = x_i[:, :x_i.shape[1]-3]
        x_j = x_j[:, :x_j.shape[1]-3]
        m_inputs = (torch.cat((x_i, x_j, edge_attr, results), dim=1))
        m_outputs = self.phi_e(m_inputs)

        info = self.phi_x(m_outputs)
        z = diff[:] * info

        return torch.cat((m_outputs, z), dim=1)

    def forward(self, x, edge_index, edge_attr, coords):
        old_x = x.clone()

        x = torch.cat((x, coords), dim=1)

        outputs = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        info = outputs[:, len(outputs[0])-3:]
        outputs = outputs[:, :len(outputs[0])-3]
        inputs = torch.cat((old_x, outputs), dim=1)
        new_coords = coords + info / len(x)
        return self.phi_h(inputs), new_coords

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
        x, edge_index, batch, coords  = data.x, data.edge_index, data.batch, data.pos

        x, coords = self.gnn1(x, edge_index, data.edge_attr, coords)
        x = self.silu(x)

        x, coords = self.gnn2(x, edge_index, data.edge_attr, coords)
        x = self.silu(x)

        x, coords = self.gnn3(x, edge_index, data.edge_attr, coords)
        x = self.silu(x)

        # x, coords = self.gnn4(x, edge_index, data.edge_attr, coords)
        # x = self.silu(x)

        # x, coords = self.gnn5(x, edge_index, data.edge_attr, coords)
        # x = self.silu(x)

        # x, coords = self.gnn6(x, edge_index, data.edge_attr, coords)
        # x = self.silu(x)

        # x, coords = self.gnn7(x, edge_index, data.edge_attr, coords)
        # x = self.silu(x)

        x = global_add_pool(x, batch)

        x = self.linear(x)
        x = self.silu(x)

        x = self.linear2(x)
        return x



