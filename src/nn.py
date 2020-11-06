import torch
from torch import nn


class Policy(nn.Module):
    """
    The policy net consists of two recurrent layers using GRUs, a fully connected layer and a softmax function so that
    the output can be seen as distribution over the possible action choices. The output will be a tensor of size
    (batch size, 1, one hot length) with each index of the one hot dimension representing a action choice.
    """

    def __init__(self):
        super(Policy, self).__init__()

        self.input_dim = 36  # onehot of current position 
        self.output_dim = 4  # action probs
        self.temperature = 1.5

        self.lin = nn.Linear(self.input_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.lin(x)
        out = self.softmax(out / self.temperature)
        return out

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=torch.device(device)))

    def set_parameters_to(self, policy):
        self.load_state_dict(policy.state_dict())
