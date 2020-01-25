import argparse

import gym
import numpy as np
from IPython.core.display import clear_output
import matplotlib.pyplot as plt

def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch SAC")

    parser.add_argument("--render", action="store_true",
                        help="Render simulation (default: False)")

    parser.add_argument("--tensorboard", action="store_true",
                        help="Use tensorboard for tracking (default: False)")

    parser.add_argument("--verbose", type=int, default=1, metavar="",
                        help="Verbose level [1..3] (default: 1)")

    parser.add_argument("--alpha", type=float, default=0.2, metavar="",
                        help="Temperature parameter α determines the relative importance of "
                             "the entropy term against the reward (default: 0.2)")

    parser.add_argument("--gamma", type=float, default=0.99, metavar="",
                        help="Discount factor for reward (default: 0.99)")

    parser.add_argument("--tau", type=float, default=0.005, metavar="",
                        help="Target smoothing coefficient (default: 0.005)")

    parser.add_argument("--lr", type=float, default=0.0003, metavar="",
                        help="Learning rate (default: 3e-4)")

    parser.add_argument("--hidden_units", type=int, default=256, metavar="",
                        help="Number of units in hidden layers (default: 256)")

    parser.add_argument("--replay_size", type=int, default=1000000, metavar="",
                        help="Size of replay buffer (default: 1e6)")

    parser.add_argument("--seed", type=int, default=0, metavar="",
                        help="Random seed (default: 0)")

    parser.add_argument("--batch_size", type=int, default=256, metavar="",
                        help="Batch size (default: 256)")

    parser.add_argument("--initial_epsilon", type=float, default=None, metavar="",
                        help="Initial value of epsilon, which is the probability of"
                             "random sampling environment actions (default: None) "
                             "[must be set along with --epsilon_decrease]")

    parser.add_argument("--epsilon_decay", type=float, default=None, metavar="",
                        help="Exponential decay of epsilon (default: None)"
                             "[must be set along with --initial_epsilon]")

    parser.add_argument("--epsilon_decrease", type=float, default=None, metavar="",
                        help="Linear decrease of epsilon (default: None)"
                             "[must be set along with --initial_epsilon]")

    parser.add_argument("--final_epsilon", type=float, default=None, metavar="",
                        help="Final value of epsilon (default: None) "
                             "[must be set along with --epsilon_decrease]")

    parser.add_argument("--max_steps", type=int, default=1000000, metavar="",
                        help="Maximum number of timesteps (default: 1e6)")

    parser.add_argument("--max_episode_steps", type=int, default=1000, metavar="",
                        help="Maximum number of timesteps per episode (default: 1000)")

    parser.add_argument("--exploratory_steps", type=int, default=None, metavar="",
                        help="Number of exploratory (i.e. with random actions) "
                             "initial steps (default: None)")

    parser.add_argument("--gradient_steps", type=int, default=1, metavar="",
                        help="Gradient steps per simulator step (default: 1)")

    parser.add_argument("--target_update_interval", type=int, default=1, metavar="",
                        help="Value target update per number of updates per step (default: 1)")

    parser.add_argument("--save_params_interval", type=int, default=None, metavar="",
                        help="If set, interval of episodes to save net params (default: None)")

    parser.add_argument("--load_params", type=str, default=None, metavar="",
                        help="Dir of the three NNs parameters (default: None)")

    return parser


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

def plot(i_episode, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Episode %s. Reward: %s' % (i_episode, rewards[-1]))
    plt.plot(rewards)
    plt.show()