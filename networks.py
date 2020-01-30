import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# to ensure stability of neural networks learning
MAX_LOG_STD = 2
MIN_LOG_STD = -20
EPS = 1e-6
# absolute bounds for weights initialization
INIT_WEIGHT = 1E-2


class ValueNetwork(nn.Module):
    """
    A class that implements the neural network for the parametrized function approximator
    for the value function
    It inherits from PyTorch nn.Module and thus has to implement the abstract method forward that
    computes an output given an input
    The architecture is simple:
        - feedforward fully connected
        - input layer of dimension of the observation space, to process state
        - just two hidden layers with ReLU activation function
        - linear activation function at the output layer

    It is not actually used since in the second SAC paper it was shown to be redundant
    """
    def __init__(self, state_dim, hidden_dim):
        """
        Constructor
        :param state_dim: dimension of the observation space
        :type state_dim: int
        :param hidden_dim: dimension of the hidden layers
        :type hidden_dim: int
        """
        super().__init__()

        # build layers
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)                     # out = V(s)

        # initialize all layers parameters
        self.linear1.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear1.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear2.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear3.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear3.bias.data.uniform_(0, INIT_WEIGHT)

    def forward(self, state):
        """
        Implements the PyTorch nn.Module abstract method to compute the output of a neural network given an input
        In this case it computes the value of a given state
        :param state: state
        :return: estimated value of state given the current parameters
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    """
    A class that implements the neural network for the parametrized function approximator
    for the Q function
    It inherits from PyTorch nn.Module and thus has to implement the abstract method forward that
    computes an output given an input
    The architecture is simple:
        - input layer of dimension of the observation space and of the action space,
            to process state-action pair
        - feedforward fully connected
        - just two hidden layers with ReLU activation function
        - linear activation function at the output layer
    """
    def __init__(self, state_dim, action_dim, hidden_size):
        """
        Constructor
        :param state_dim: dimension of environment observation space
        :type state_dim: int
        :param action_dim: dimension of environment action space
        :type action_dim: int
        :param hidden_size: dimension of the hidden layers
        :type hidden_size: int
        """
        super().__init__()

        # build layers
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)                            # out = Q(s,a)

        # initialize parameters (bias uniform from 0 to avoid ReLU blocking gradient from the start)
        self.linear1.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear1.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear2.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear3.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear3.bias.data.uniform_(0, INIT_WEIGHT)

    def forward(self, state, action):
        """
        Implements the PyTorch nn.Module abstract method to compute the output of a neural network given an input
        In this case it computes the Q value of a given state
        :param state: state
        :type state: torch float tensor
        :param action: action
        :type state: torch float tensor
        :return: estimated value of state-action pair given the current parameters
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """
    A class that implements the neural network for the parametrized function approximator
    for the policy, represented as a squashed Gaussian
    It inherits from PyTorch nn.Module and thus has to implement the abstract method forward that
    computes an output given an input
    The architecture is simple:
        - input layer of dimension of the observation space,
            to process state
        - feedforward fully connected
        - just two hidden layers with ReLU activation function
        - linear activation function at the two output layers, to compute:
            - mean of the action probability distribution
            - log std of the action probability distribution
    """
    def __init__(self, num_inputs, num_actions, hidden_size, device):
        """
        Constructor
        :param num_inputs: dimension of environment observation space
        :param num_actions: dimension of environment action space
        :param hidden_size: dimension of hidden layers
        :param device: device (cuda or cpu)
        """
        super().__init__()

        # needed to bring incoming states to correct device for computations other than PyTorch forward and backprop
        self.device = device

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)   # out = pi(a|s) = [pi(a_1|s), ..., pi(a_n|s)]

        self.linear1.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear1.bias.data.uniform_(0, INIT_WEIGHT)
        self.linear2.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.linear2.bias.data.uniform_(0, INIT_WEIGHT)

        self.mean_linear.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.mean_linear.bias.data.uniform_(0, INIT_WEIGHT)

        # log std helps loss backprop, can recover std with exponentiation later
        self.log_std_linear.weight.data.uniform_(-INIT_WEIGHT, INIT_WEIGHT)
        self.log_std_linear.bias.data.uniform_(0, INIT_WEIGHT)

    def forward(self, state):
        """
        Implements the PyTorch nn.Module abstract method to compute the output of a neural network given an input
        In this case it computes the action distribution in a given state (parametrized as a gaussian)
        :param state: state
        :type state: torch float tensor
        :return: estimated mean, log std of the gaussian distribution of actions in given state
        :rtype: tuple of torch tensors
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # clamp the log std to avoid numerical instability
        log_std = torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD)

        return mean, log_std

    def sample(self, state):
        """
        Samples the actions probability for the given state
        Used in training
        :param state: state
        :type state: torch float tensor
        :return: action, log pi(a|s), unsquashed action, gaussian mean, gaussian log std
        :rtype: tuple of torch tensors
        """

        # get neural network estimation for action distribution mean and log std
        mean, log_std = self.forward(state)
        # get std by exponentiation
        std = log_std.exp()

        # sample independent zero mean unit variance gaussian noise
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)

        # compute action as the squashed tanh of gaussian prediction multiplied by independent noise,
        # needed for the reparametrization trick (see Haarnoja et al. paper)
        action = torch.tanh(mean + std * z)

        # to recover normalized log probabilities (see Harnojaa et al. Appendix "Enforcing Bounds")
        log_prob = Normal(mean, std).log_prob(mean + std * z) \
                   - torch.log(1 - action.pow(2) + EPS).sum(1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        """
        Simply samples from policy distribution and returns the action
        Used to actually choose an action to perform
        :param state: state
        :type state: numpy array
        :return: action to be performed in given state according to policy
        :rtype: torch tensor
        """
        state = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float)
        action, log_prob, z, mean, log_std = self.sample(state)
        return action[0]


def hard_update(source_net, target_net):
    """
    Hard updates target network parameters by simply copying source source network parameters values
    The two networks must share the same architecture
    :param source_net: source neural network
    :param target_net: target neural network
    :return: None
    """
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)


def soft_update(source_net, target_net, tau):
    """
    Soft updates target network parameters with an exponentially moving average of
    source source network parameters values
    The two networks must share the same architecture
    :param source_net: source neural network
    :param target_net: target neural network
    :param tau: coefficient of soft update (in [0,1]: 1 -> hard update, 0 -> no update)
    :return: None
    """
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

