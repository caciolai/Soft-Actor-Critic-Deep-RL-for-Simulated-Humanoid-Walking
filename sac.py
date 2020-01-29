import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import QNetwork, PolicyNetwork, hard_update, soft_update

"""
This file contains the actual implementation of the SAC algorithm
"""

class SAC:
    """
    A class used to represent a SAC agent

    Attributes
    ----------
    device : cuda or cpu
        the device on which all the computation occurs
    gamma : float[0,1]
        discount factor
    state_dim : int
        dimension of the environment observation space
    action_dim : int
        dimension of the environment action space
    hidden_dim : int
        dimension of the hidden layers of the networks
    tau : float[0,1]
        coefficient of soft update of target networks
    lr : float
        learning rate of the optimizers
    target_update_interval : int
        number of updates in between soft updates of target networks
    q_net_1 : QNetwork
        soft Q value network 1
    q_net_2 : QNetwork
        soft Q value network 2
    target_q_net_1 : QNetwork
        target Q value network 1
    target_q_net_2 : QNetwork
        target Q value network 2
    policy_net : PolicyNetwork
        policy network
    q1_criterion :
        torch optimization criterion for q_net_1
    q2_criterion :
        torch optimization criterion for q_net_2
    q1_optim :
        torch optimizer for q_net_1
    q2_optim :
        torch optimizer for q_net_2
    policy_optim :
        torch optimizer for policy_net
    alpha : torch float scalar
        entropy temperature (controls policy stochasticity)
    entropy_target : torch float scalar
        entropy target for the environment (see Harnojaa et al. Section 5)

    Methods
    -------
    update(replay_buffer, batch_size, updates) : q1_loss, q2_loss, policy_loss, alpha_loss
         Performs a gradient step of the algorithm, optimizing Q networks and policy network and optimizing alpha

    choose_action(state) : action
        Returns the appropriate action in given state according to current policy

    save_networks_parameters(params_dir)
        Saves the relevant parameters (q1_net's, q2_net's, policy_net's, alpha) from the networks

    load_networks_parameters(params_dir)
        Loads the relevant parameters (q1_net's, q2_net's, policy_net's, alpha) into the networks

    """
    def __init__(self, observation_space, action_space, args):
        """
        Constructor
        :param observation_space: observation space of the environment
        :param action_space: action space of the environment
        :param args: command line args to set hyperparameters
        """

        # set hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.hidden_dim = args.hidden_units
        self.tau = args.tau
        self.lr = args.lr
        self.target_update_interval = args.target_update_interval

        # build and initialize networks
        self.q_net_1 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.q_net_2 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_q_net_1 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_q_net_2 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        hard_update(self.q_net_1, self.target_q_net_1)
        hard_update(self.q_net_2, self.target_q_net_2)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(self.device)

        # build criterions and optimizers
        self.q1_criterion = nn.MSELoss()
        self.q2_criterion = nn.MSELoss()
        self.q1_optim = optim.Adam(self.q_net_1.parameters(), lr=self.lr)
        self.q2_optim = optim.Adam(self.q_net_2.parameters(), lr=self.lr)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # for optimizing alpha (see Harnojaa et al. section 5)
        if args.initial_alpha is not None:
            self.alpha = torch.tensor(args.initial_alpha, requires_grad=True, device=self.device, dtype=torch.float)
        else:
            self.alpha = torch.rand(1, requires_grad=True, device=self.device, dtype=torch.float)

        if args.entropy_target is not None:
            self.entropy_target = torch.tensor(args.target_alpha, device=self.device, dtype=torch.float)
        else:
            self.entropy_target = -1. * torch.tensor(action_space.shape, device=self.device, dtype=torch.float)

        self.alpha_optim = optim.Adam([self.alpha], lr=self.lr)

    def update(self, replay_buffer, batch_size, updates):
        """
        Performs a gradient step of the algorithm, optimizing Q networks and policy network and optimizing alpha
        :param replay_buffer: replay buffer to sample batches of transitions from
        :param batch_size: size of the batches
        :param updates: number of updates so far
        :return: losses of the four optimizers (q1_optim, q2_optim, policy_optim, alpha_optim)
        :rtype: tuple of torch scalar floats
        """

        # sample a transition batch from replay buffer and cast it to tensor of the correct shape
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        state_batch = torch.from_numpy(state_batch).to(self.device, dtype=torch.float)
        next_state_batch = torch.from_numpy(next_state_batch).to(self.device, dtype=torch.float)
        action_batch = torch.from_numpy(action_batch).to(self.device, dtype=torch.float)
        reward_batch = torch.from_numpy(reward_batch).unsqueeze(1).to(self.device, dtype=torch.float)
        done_batch = torch.from_numpy(np.float32(done_batch)).unsqueeze(1).to(self.device, dtype=torch.float)

        # sample actions from the policy to be used for expectations updates
        sampled_action, log_prob, epsilon, mean, log_std = self.policy_net.sample(state_batch)

        ### evaluation step
        target_next_value = torch.min(
            self.target_q_net_1(next_state_batch, sampled_action),
            self.target_q_net_2(next_state_batch, sampled_action)
        ) - self.alpha * log_prob

        current_q_value_1 = self.q_net_1(state_batch, action_batch)
        current_q_value_2 = self.q_net_2(state_batch, action_batch)

        expected_next_value = reward_batch + (1 - done_batch) * self.gamma * target_next_value
        q1_loss = self.q1_criterion(current_q_value_1, expected_next_value.detach())
        q2_loss = self.q2_criterion(current_q_value_2, expected_next_value.detach())

        # optimize q1 and q1 nets
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        ### improvement step
        sampled_q_value = torch.min(
            self.q_net_1(state_batch, sampled_action),
            self.q_net_2(state_batch, sampled_action)
        )
        policy_loss = (self.alpha * log_prob - sampled_q_value).mean()

        # optimize policy net
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # optimize alpha
        alpha_loss = (self.alpha * (-log_prob - self.entropy_target).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # update Q target value
        if updates % self.target_update_interval == 0:
            soft_update(self.q_net_1, self.target_q_net_1, self.tau)
            soft_update(self.q_net_2, self.target_q_net_2, self.tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()

    def choose_action(self, state):
        """
        Returns the appropriate action in given state according to current policy
        :param state: state
        :return: action
        :rtype numpy float array
        """

        action = self.policy_net.get_action(state)
        # move to cpu, remove from gradient graph, cast to numpy
        return action.cpu().detach().numpy()

    def save_networks_parameters(self, params_dir=None):
        """
        Saves the relevant parameters (q1_net's, q2_net's, policy_net's, alpha) from the networks
        :param params_dir: directory where to save parameters to (optional)
        :return: None
        """
        if params_dir is None:
            params_dir = "SavedAgents/"

        # create a subfolder with current timestamp
        prefix = os.path.join(params_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        policy_path = os.path.join(prefix, "policy_net_params")
        q1_path = os.path.join(prefix, "q1_net_params")
        q2_path = os.path.join(prefix, "q2_net_params")
        alpha_path = os.path.join(prefix, "alpha_param")

        print("Saving parameters to {}, {}, {}".format(q1_path, q2_path, policy_path))

        torch.save(self.q_net_1.state_dict(), q1_path)
        torch.save(self.q_net_2.state_dict(), q2_path)
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.alpha, alpha_path)

        return params_dir

    def load_networks_parameters(self, params_dir):
        """
        Loads the relevant parameters (q1_net's, q2_net's, policy_net's, alpha) into the networks
        :param params_dir: directory where to load parameters from
        :return: None
        """
        if params_dir is not None:
            print("Loading parameters from {}".format(params_dir))

            policy_path = os.path.join(params_dir, "policy_net_params")
            self.policy_net.load_state_dict(torch.load(policy_path))

            q1_path = os.path.join(params_dir, "q1_net_params")
            q2_path = os.path.join(params_dir, "q2_net_params")
            self.q_net_1.load_state_dict(torch.load(q1_path))
            self.q_net_2.load_state_dict(torch.load(q2_path))

            alpha_path = os.path.join(params_dir, "alpha_param")
            self.alpha = torch.load(alpha_path)

