import os
import datetime
from torch.optim import Adam
from networks import *


class Agent:
    def __init__(self, num_inputs, action_space, args):
        # set hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build policy network
        self.policy = PolicyNetwork(num_inputs, action_space.shape[0], args.hidden_units, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # build Q1 and Q2 networks
        self.Q1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_units).to(self.device)
        self.Q2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_units).to(self.device)
        self.Q1_optim = Adam(self.Q1.parameters(), lr=args.lr)
        self.Q2_optim = Adam(self.Q2.parameters(), lr=args.lr)

        # build value network (and target value network)
        self.value = ValueNetwork(num_inputs, action_space.shape[0], args.hidden_units).to(self.device)
        self.value_target = ValueNetwork(num_inputs, action_space.shape[0], args.hidden_units).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=args.lr)

        # initialize target value net parameters equal to value net parameters
        hard_update(self.value_target, self.value)


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _, _ = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return action


    def update_networks_parameters(self, replay_buffer, minibatch_size, updates):
        # sample a minibatch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(
            batch_size=minibatch_size
        )
        # cast and move batches to GPU
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # Jq = E_{(s,a)~D}[ ( Q(s,a) - Q'(s,a) )^2 ]
        # where Q'(s,a) = r + gamma * (1-d) * V_targ (s')

        # compute Q' (based on value target params updated separately so no need to include computation graph)
        with torch.no_grad():
            V_targ = self.value_target(next_state_batch)
            Q_hat = reward_batch + done_batch * (1 - self.gamma) * V_targ

        # compute Q functions on state action pairs from buffer
        q1_D = self.Q1(state_batch, action_batch)
        q2_D = self.Q2(state_batch, action_batch)

        # compute both soft Q functions loss
        Jq1 = F.mse_loss(q1_D, Q_hat)
        Jq2 = F.mse_loss(q2_D, Q_hat)

        # Jp = E_{(s~D), (a~pi)}[ ( log pi(a | s) - Q(s, a) )^2 ]
        # with reparametrization trick to express it in terms of noise
        # and also minimum of the two Q(s,a)

        # sample actions from policy and compute Q functions with these sampled actions
        pi, log_pi, mean, log_std = self.policy.sample(state_batch)
        q1_pi = self.Q1(state_batch, pi)
        q2_pi = self.Q2(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        Jp = ((self.alpha * log_pi) - min_q_pi).mean()

        # Jv = E_{s~D}[( V(s) - E_{a~pi}[Q(s,a) - log pi(a|s)] )^2]
        # using sampled actions
        # and also minimum of the two Q(s,a)

        vf = self.value(state_batch)
        # again, since value target, no need to include computation graph for gradient
        with torch.no_grad():
            vf_target = min_q_pi - (self.alpha * log_pi)

        Jv = F.mse_loss(vf, vf_target)

        # update all parameters by one step SGD
        self.value_optim.zero_grad()
        Jv.backward()
        self.value_optim.step()

        self.Q1_optim.zero_grad()
        Jq1.backward()
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        Jq2.backward()
        self.Q2_optim.step()

        self.policy_optim.zero_grad()
        Jp.backward()
        self.policy_optim.step()

        # update target value function
        if updates % self.target_update_interval == 0:
            soft_update(self.value_target, self.value, self.tau)

        # return losses for tracking
        return Jv.item(), Jq1.item(), Jq2.item(), Jp.item()


    def save_networks_parameters(self):
        prefix = "SavedAgents/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        policy_path = prefix + "policy_net_params"
        q1_path = prefix + "q1_net_params"
        q2_path = prefix + "q2_net_params"
        value_path = prefix + "value_net_params"

        print("Saving parameters to {}, {}, {} and {}".format(policy_path, q1_path, q2_path, value_path))

        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.Q1.state_dict(), q1_path)
        torch.save(self.Q2.state_dict(), q2_path)
        torch.save(self.value.state_dict(), value_path)


    def load_networks_parameters(self, policy_path, q1_path, q2_path, value_path):
        if policy_path is not None:
            print("Loading parameters for policy from {}".format(policy_path))
            self.policy.load_state_dict(torch.load(policy_path))
        if q1_path is not None:
            print("Loading parameters for Q functions from {}".format(q1_path))
            self.Q1.load_state_dict(torch.load(q1_path))
        if q2_path is not None:
            print("Loading parameters for Q functions from {}".format(q2_path))
            self.Q1.load_state_dict(torch.load(q2_path))
        if value_path is not None:
            print("Loading parameters for value function from {}".format(value_path))
            self.value.load_state.dict(torch.load(value_path))