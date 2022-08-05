from torch import nn

class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True):
        super().__init__()
        if activation:
            self.layers = nn.Sequential(nn.Linear(in_channel, out_channel),
            nn.ReLU())
            
        else:
            self.layers = nn.Sequential(nn.Linear(in_channel, out_channel))


    def forward(self, x):
        return self.layers(x)