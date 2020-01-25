import os
import datetime

import numpy as np
import torch.optim as optim
from networks import *


class SAC:
    def __init__(self, observation_space, action_space, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.action_dim = action_space.shape[0]
        self.state_dim = observation_space.shape[0]
        self.hidden_dim = args.hidden_units
        self.tau = args.tau
        self.lr = args.lr

        self.value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        hard_update(self.value_net, self.target_value_net)
        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(self.device)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)


    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Value Function
        predicted_new_q_value = torch.min(
            self.soft_q_net1(state, new_action),
            self.soft_q_net2(state, new_action)
        )
        target_value_func = predicted_new_q_value - log_prob
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update Target Value
        soft_update(self.value_net, self.target_value_net, self.tau)

    def save_networks_parameters(self, prefix=None):
        if not prefix:
            prefix = "SavedAgents/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        policy_path = prefix + "/" + "policy_net_params"
        q1_path = prefix + "/" + "q1_net_params"
        q2_path = prefix + "/" + "q2_net_params"
        value_path = prefix + "/" + "value_net_params"

        print("Saving parameters to {}, {}, {} and {}".format(policy_path, q1_path, q2_path, value_path))

        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.soft_q_net1.state_dict(), q1_path)
        torch.save(self.soft_q_net2.state_dict(), q2_path)
        torch.save(self.value_net.state_dict(), value_path)

    def load_networks_parameters(self, params_path):
        if params_path is not None:
            print("Loading parameters from {}".format(params_path))

            policy_path = params_path + "/" + "policy_net_params"
            self.policy_net.load_state_dict(torch.load(policy_path))

            q1_path = params_path + "/" + "q1_net_params"
            q2_path = params_path + "/" + "q2_net_params"
            self.soft_q_net1.load_state_dict(torch.load(q1_path))
            self.soft_q_net2.load_state_dict(torch.load(q2_path))

            value_path = params_path + "/" + "value_net_params"
            self.value_net.load_state_dict(torch.load(value_path))