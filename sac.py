import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import QNetwork, PolicyNetwork, hard_update, soft_update


class SAC:
    def __init__(self, observation_space, action_space, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.action_dim = action_space.shape[0]
        self.state_dim = observation_space.shape[0]
        self.hidden_dim = args.hidden_units
        self.tau = args.tau
        self.lr = args.lr
        self.target_update_interval = args.target_update_interval

        # self.value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        # self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        # hard_update(self.value_net, self.target_value_net)
        self.q_net1 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.q_net2 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_q_net1 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_q_net2 = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        hard_update(self.q_net1, self.target_q_net1)
        hard_update(self.q_net2, self.target_q_net2)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(self.device)

        # self.value_criterion = nn.MSELoss()
        self.q1_criterion = nn.MSELoss()
        self.q2_criterion = nn.MSELoss()

        # self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.q1_optim = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q2_optim = optim.Adam(self.q_net2.parameters(), lr=self.lr)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # for optimizing alpha
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
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        state_batch = torch.from_numpy(state_batch).to(self.device, dtype=torch.float)
        next_state_batch = torch.from_numpy(next_state_batch).to(self.device, dtype=torch.float)
        action_batch = torch.from_numpy(action_batch).to(self.device, dtype=torch.float)
        reward_batch = torch.from_numpy(reward_batch).unsqueeze(1).to(self.device, dtype=torch.float)
        done_batch = torch.from_numpy(np.float32(done_batch)).unsqueeze(1).to(self.device, dtype=torch.float)

        current_value1 = self.q_net1(state_batch, action_batch)
        current_value2 = self.q_net2(state_batch, action_batch)
        # predicted_value = self.value_net(state)
        sampled_action, log_prob, epsilon, mean, log_std = self.policy_net.sample(state_batch)

        alpha = self.alpha

        # Training Q Function
        target_next_value = torch.min(
            self.target_q_net1(next_state_batch, sampled_action),
            self.target_q_net2(next_state_batch, sampled_action)
        ) - alpha * log_prob

        # target_next_value = self.target_value_net(next_state)
        expected_next_value = reward_batch + (1 - done_batch) * self.gamma * target_next_value
        q1_loss = self.q1_criterion(current_value1, expected_next_value.detach())
        q2_loss = self.q2_criterion(current_value2, expected_next_value.detach())

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # # Training Value Function
        # sampled_q_value = torch.min(
        #     self.soft_q_net1(state, sampled_action),
        #     self.soft_q_net2(state, sampled_action)
        # )
        # sampled_target_value = sampled_q_value - log_prob
        # value_loss = self.value_criterion(predicted_value, sampled_target_value.detach())
        #
        # self.value_optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optimizer.step()

        # Training Policy Function
        sampled_q_value = torch.min(
            self.q_net1(state_batch, sampled_action),
            self.q_net2(state_batch, sampled_action)
        )
        policy_loss = (alpha * log_prob - sampled_q_value).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Optimizing alpha
        alpha_loss = (self.alpha * (-log_prob - self.entropy_target).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # # Update Target Value
        # if updates % self.target_update_interval == 0:
        #     soft_update(self.value_net, self.target_value_net, self.tau)

        # Update Q Target Value
        if updates % self.target_update_interval == 0:
            soft_update(self.q_net1, self.target_q_net1, self.tau)
            soft_update(self.q_net2, self.target_q_net2, self.tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()


    def choose_action(self, state):
        action = self.policy_net.get_action(state)
        return action.detach().numpy()

    def save_networks_parameters(self, params_dir=None):
        if params_dir is None:
            params_dir = "SavedAgents/"

        prefix = os.path.join(params_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        policy_path = os.path.join(prefix, "policy_net_params")
        q1_path = os.path.join(prefix, "q1_net_params")
        q2_path = os.path.join(prefix, "q2_net_params")
        # value_path = prefix + "value_net_params"
        alpha_path = os.path.join(prefix, "alpha_param")

        # print("Saving parameters to {}, {}, {} and {}".format(policy_path, q1_path, q2_path, value_path))
        print("Saving parameters to {}, {}, {}".format(q1_path, q2_path, policy_path))

        torch.save(self.q_net1.state_dict(), q1_path)
        torch.save(self.q_net2.state_dict(), q2_path)
        torch.save(self.policy_net.state_dict(), policy_path)
        # torch.save(self.value_net.state_dict(), value_path)
        torch.save(self.alpha, alpha_path)

        return params_dir

    def load_networks_parameters(self, params_dir):
        if params_dir is not None:
            print("Loading parameters from {}".format(params_dir))

            policy_path = os.path.join(params_dir, "policy_net_params")
            self.policy_net.load_state_dict(torch.load(policy_path))

            q1_path = os.path.join(params_dir, "q1_net_params")
            q2_path = os.path.join(params_dir, "q2_net_params")
            self.q_net1.load_state_dict(torch.load(q1_path))
            self.q_net2.load_state_dict(torch.load(q2_path))

            # value_path = params_path + "/" + "value_net_params"
            # self.value_net.load_state_dict(torch.load(value_path))

            alpha_path = os.path.join(params_dir, "alpha_param")
            self.alpha = torch.load(alpha_path)

