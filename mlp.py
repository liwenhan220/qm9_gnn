from torch import nn

class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True):
        super().__init__()
        if activation:
            self.layers = nn.Sequential(nn.Linear(in_channel, out_channel),
            nn.ReLU())
            # nn.Linear(16, out_channel),
            # nn.SiLU())
        else:
            self.layers = nn.Sequential(nn.Linear(in_channel, out_channel))
            # nn.SiLU(),
            # nn.Linear(16, out_channel))


    def forward(self, x):
        return self.layers(x)