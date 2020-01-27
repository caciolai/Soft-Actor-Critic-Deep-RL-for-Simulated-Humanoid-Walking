import argparse

import gym
from gym import spaces
import numpy as np
import sklearn, sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch SAC")

    parser.add_argument("--render", action="store_true",
                        help="Render simulation (default: False)")

    parser.add_argument("--verbose", type=int, default=1, metavar="",
                        help="Verbose level [0..3] (default: 1)")

    parser.add_argument("--initial_alpha", type=float, default=None, metavar="",
                        help="Initial value for alpha (default: None -> 0)")

    parser.add_argument("--entropy_target", type=float, default=None, metavar="",
                        help="Entropy target for alpha update (default: None -> -dim(A))")

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

    parser.add_argument("--batch_size", type=int, default=256, metavar="",
                        help="Batch size (default: 256)")

    parser.add_argument("--seed", type=int, default=0, metavar="",
                        help="Random seed (default: 0)")

    parser.add_argument("--initial_epsilon", type=float, default=None, metavar="",
                        help="Initial value of epsilon, which is the probability of "
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

    parser.add_argument("--learning_starts", type=int, default=0, metavar="",
                        help="How many steps of the model to collect transitions for "
                             "before learning starts (default: 0)")

    parser.add_argument("--max_episodes", type=int, default=None, metavar="",
                        help="Maximum number of episodes (default: None)")

    parser.add_argument("--max_steps", type=int, default=None, metavar="",
                        help="Maximum number of timesteps (default: None)")

    parser.add_argument("--max_episode_steps", type=int, default=None, metavar="",
                        help="Maximum number of timesteps per episode (default: None "
                             "[Environment dependent])")

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
                        help="Directory with the neural network parameters to be loaded (default: None)")

    parser.add_argument("--testing", action="store_true",
                        help="Make a test after training")

    parser.add_argument("--testing_steps", type=int, default=1000, metavar="",
                        help="Number of testing steps (default 1000) [need --testing]")

    parser.add_argument("--plot", action="store_true",
                        help="Plot reward curve (default: False)")

    parser.add_argument("--plot_interval", type=int, default=1, metavar="",
                        help="Number of episodes between plots (default: 1)")

    return parser


def smooth(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_data(data, title, x_label, y_label, smoothness=0.6):
    plt.clf()
    smoothed_data = smooth(data, smoothness)
    plt.plot(np.arange(1, len(data)+1), smoothed_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.ion()
    plt.draw()
    plt.pause(0.1)

def plot_episodes_reward(episodes_reward_list):
    title = "Reward per episode"
    x_label = "Episode"
    y_label = "Reward"
    plot_data(episodes_reward_list, title, x_label, y_label)



# action in [-1, 1] to action in [low, high]
class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = ((high + low) + action*(high - low)) / 2.0
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = (2.0 * action - (high + low)) / (high - low)
        action = np.clip(action, -1, 1)

        return action

    def get_max_episode_steps(self):
        return self.env._max_episode_steps

    def set_max_episode_steps(self, num):
        self.env._max_episode_steps = num


class FeaturizedStates(gym.ObservationWrapper):
    def __init__(self, env, n_components=100):
        super().__init__(env)
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        obs_dim = env.observation_space.shape[0]
        features = []
        variances = np.geomspace(0.1, 10, num=obs_dim)
        for i in range(1, obs_dim+1):
            features.append(
                ("rbf{}".format(i), RBFSampler(gamma=variances[i-1], n_components=n_components))
            )

        self.featurizer = sklearn.pipeline.FeatureUnion(features)
        self.featurizer.fit(self.scaler.transform(observation_examples))

        new_obs_dim = obs_dim * n_components
        self.observation_space = spaces.Box(low=np.array([-1.0 for _ in range(new_obs_dim)]),
                                            high=np.array([1.0 for _ in range(new_obs_dim)]),
                                            dtype=np.float32)

    def observation(self, observation):
        scaled = self.scaler.transform([observation])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


