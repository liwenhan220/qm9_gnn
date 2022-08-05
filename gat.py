from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, hidden_channels=128, out_features=1):
        super(NN, self).__init__()
        self.gat1 = GATv2Conv(11, hidden_channels, edge_dim=4)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=4)
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=4)
        self.gat4 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=4)
        self.gat5 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=4)
        self.gat6 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=4)
        self.gat7 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=4)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_channels, 64)
        self.linear2 = nn.Linear(64, out_features)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.gat1(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.relu(x)

        # x = self.gat4(x, edge_index, edge_attr)
        # x = self.relu(x)

        # x = self.gat5(x, edge_index, edge_attr)
        # x = self.relu(x)

        # x = self.gat6(x, edge_index, edge_attr)
        # x = self.relu(x)

        # x = self.gat7(x, edge_index, edge_attr)
        # x = self.relu(x)

        x = global_add_pool(x, batch)

        x = self.linear(x)
        x = self.relu(x)

        x = self.linear2(x)
        return x
