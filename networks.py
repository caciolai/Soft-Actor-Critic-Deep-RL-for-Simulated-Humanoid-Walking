import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

MAX_LOG_STD = 20
MIN_LOG_STD = -20
EPSILON = 1e-6

# update target network parameters with soft update (exponentially moving average)
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * target_param.data + (1.0 - tau) * target_param)

# update target network parameters with hard update (just copy)
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# initialize parameters
def initialize_parameters(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain=1)
        torch.nn.init.constant_(layer.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, device):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(state_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, action_dim)

        self.apply(initialize_parameters)

    def forward(self, state):
        h1 = F.relu(self.linear1(state))
        h2 = F.relu(self.linear2(h1))
        v = self.linear3(h2)

        return v


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, device):
        super().__init__()

        self.device = device
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, action_dim)

        self.apply(initialize_parameters)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        q = self.linear3(h2)

        return q


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, device):
        super().__init__()

        self.device = device
        self.linear1 = nn.Linear(state_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)

        self.mean_linear = nn.Linear(hidden_units, action_dim)
        self.log_std_linear = nn.Linear(hidden_units, action_dim)

        self.apply(initialize_parameters)

    def forward(self, state):
        h1 = F.relu(self.linear1(state))
        h2 = F.relu(self.linear2(h1))
        mean = self.mean_linear(h2)
        log_std = self.log_std_linear(h2)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        noise = normal.sample().to(self.device)
        a_tilde = mean + std*noise
        action = torch.tanh(a_tilde)
        log_prob = Normal(mean, std).log_prob(a_tilde) - torch.log(1 - action.pow(2) + EPSILON)

        return action, log_prob, mean, std
