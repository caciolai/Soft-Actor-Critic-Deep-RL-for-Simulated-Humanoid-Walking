import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# update target network parameters with soft update (exponentially moving average)
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

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
    def __init__(self, state_dim, action_dim, hidden_units):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, action_dim)

        self.apply(initialize_parameters)

    def forward(self, state):
        x = state
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        v = self.linear3(h2)

        return v


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super().__init__()

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
    def __init__(self, state_dim, action_dim, hidden_units, action_space):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)

        self.mean_linear = nn.Linear(hidden_units, action_dim)
        self.log_std_linear = nn.Linear(hidden_units, action_dim)

        self.apply(initialize_parameters)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # for reparametrization trick (mean + std * N(0,1))
        a_tilde = normal.rsample()
        action = torch.tanh(a_tilde) * self.action_scale + self.action_bias
        log_prob = normal.log_prob(a_tilde)

        # enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + self.action_bias)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        std = mean = torch.tanh(std) * self.action_scale + self.action_bias
        return action, log_prob, mean, std

    def to(self, *args, **kwargs):
        device = args[0]
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
