import os
import datetime
from torch.optim import Adam
from networks import *


class Agent:
    def __init__(self, num_inputs, action_space, args):
        # set hyperparameters
        self.action_space = action_space
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build policy network
        self.policy = PolicyNetwork(num_inputs, action_space.shape[0], args.hidden_units, self.device).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.lr)

        # build Q1 and Q2 networks
        self.Q1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_units, self.device).to(self.device)
        self.Q2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_units, self.device).to(self.device)
        self.Q1_optimizer = Adam(self.Q1.parameters(), lr=args.lr)
        self.Q1_criterion = nn.MSELoss()
        self.Q2_optimizer = Adam(self.Q2.parameters(), lr=args.lr)
        self.Q2_criterion = nn.MSELoss()

        # build value network (and target value network)
        self.value = ValueNetwork(num_inputs, action_space.shape[0], args.hidden_units, self.device).to(self.device)
        self.value_target = ValueNetwork(num_inputs, action_space.shape[0], args.hidden_units, self.device).to(self.device)
        self.value_optimizer = Adam(self.value.parameters(), lr=args.lr)
        self.value_criterion = nn.MSELoss()

        # initialize target value net parameters equal to value net parameters
        hard_update(self.value_target, self.value)


    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _, _, _ = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return self.rescale_action(action)


    def rescale_action(self, action):
        return action * (self.action_space.high - self.action_space.low) / 2.0 + \
               (self.action_space.high + self.action_space.low) / 2.0


    def update_networks_parameters(self, replay_buffer, minibatch_size, updates):
        # sample a minibatch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(
            batch_size=minibatch_size
        )
        # cast and move batches to GPU
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().unsqueeze(1).to(self.device)
        done_batch = torch.from_numpy(done_batch).float().unsqueeze(1).to(self.device)

        predicted_q1_value = self.Q1(state_batch, action_batch)
        predicted_q2_value = self.Q2(state_batch, action_batch)
        predicted_value = self.value(state_batch)
        sampled_action_batch, log_prob_batch, _, _ = self.policy.sample(state_batch)

        # Training Q Function
        next_target_value = self.value_target(next_state_batch)
        q_target_value = reward_batch + (torch.tensor(1.) - done_batch) * self.gamma * next_target_value
        q1_loss = self.Q1_criterion(predicted_q1_value, q_target_value.detach())
        q2_loss =  self.Q2_criterion(predicted_q2_value, q_target_value.detach())

        self.Q1_optimizer.zero_grad()
        q1_loss.backward()
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        q2_loss.backward()
        self.Q2_optimizer.step()

        # Training Value Function
        sampled_q_value = torch.min(self.Q1(state_batch, sampled_action_batch), self.Q2(state_batch, sampled_action_batch))
        predicted_target_value = sampled_q_value - log_prob_batch
        value_loss = self.value_criterion(predicted_value, predicted_target_value.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Training Policy Function
        policy_loss = (log_prob_batch - sampled_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target value function
        if updates % self.target_update_interval == 0:
            soft_update(self.value_target, self.value, self.tau)

        # return losses for tracking
        return q1_loss.item(), q2_loss.item(), value_loss.item(), policy_loss.item()


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

        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.Q1.state_dict(), q1_path)
        torch.save(self.Q2.state_dict(), q2_path)
        torch.save(self.value.state_dict(), value_path)


    def load_networks_parameters(self, params_path):
        if params_path is not None:
            print("Loading parameters from {}".format(params_path))

            policy_path = ("/" + "policy_net_params")
            self.policy.load_state_dict(torch.load(policy_path))

            q1_path = params_path + ("/" + "q1_net_params")
            q2_path = params_path + ("/" + "q2_net_params")
            self.Q1.load_state_dict(torch.load(q1_path))
            self.Q2.load_state_dict(torch.load(q2_path))

            policy_path += ("/" + "q1_net_params")
            self.value.load_state_dict(torch.load(params_path))