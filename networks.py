import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

MAX_LOG_STD = 2
MIN_LOG_STD = -20
EPS = 1e-6
INIT_WEIGHT = 1E-2


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear1.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear1.bias.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.bias.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear3.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear3.bias.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)


        self.linear1.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear1.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear2.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear3.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear3.bias.data.uniform_(0, INIT_WEIGHT)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device):
        super().__init__()

        self.device = device

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.mean_linear.bias.data.uniform_(0, INIT_WEIGHT)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.log_std_linear.bias.data.uniform_(0, INIT_WEIGHT)

        self.linear1.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear1.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear2.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.bias.data.uniform_(0, INIT_WEIGHT)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = Normal(mean, std).log_prob(
            mean + std * z.to(self.device)
        ) - torch.log(1 - action.pow(2) + EPS)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


def hard_update(source_net, target_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)

def soft_update(source_net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

