import environment
import torch
from torch import nn


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.input_dim = 4  # onehot of possible paths
        self.output_dim = 4  # action probs
        self.hidden_dim = 32
        self.layers = 2
        self.temperature = 1.2

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layers, batch_first=True)
        self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h=None):
        if h is not None:
            out, h = self.gru(x, h)

        else:
            out, h = self.gru(x)

        out = out[:, -1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out / self.temperature)

        return out, h

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=torch.device(device)))

    def set_parameters_to(self, policy):
        self.load_state_dict(policy.state_dict())
