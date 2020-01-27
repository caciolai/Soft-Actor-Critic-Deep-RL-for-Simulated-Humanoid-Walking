import argparse
import datetime
import os

import gym
from gym import spaces
import numpy as np
import sklearn, sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import matplotlib as mlp
import matplotlib.pyplot as plt

import skopt
import joblib

from simulator import train
from sac import SAC


ARGS = None
ENV = None


def build_argsparser():
    parser = argparse.ArgumentParser(description="PyTorch SAC")

    parser.add_argument("--initial_alpha", type=float, default=None, metavar="",
                        help="Initial value for alpha (default: None -> 0)")

    parser.add_argument("--entropy_target", type=float, default=None, metavar="",
                        help="Entropy target for alpha update (default: None -> -dim(A))")

    parser.add_argument("--gamma", type=float, default=0.99, metavar="",
                        help="Discount factor for reward (default: 0.99)")

    parser.add_argument("--tau", type=float, default=0.005, metavar="",
                        help="Target smoothing coefficient (default: 0.005)")

    parser.add_argument("--lr", type=float, default=0.0003, metavar="",
                        help="Learning rate (default: 0.0003)")

    parser.add_argument("--hidden_units", type=int, default=256, metavar="",
                        help="Number of units in hidden layers (default: 256)")

    parser.add_argument("--replay_size", type=int, default=int(1e6), metavar="",
                        help="Size of replay buffer (default: 1e6)")

    parser.add_argument("--batch_size", type=int, default=256, metavar="",
                        help="Batch size (default: 256)")

    parser.add_argument("--gradient_steps", type=int, default=1, metavar="",
                        help="Gradient steps per simulator step (default: 1)")

    parser.add_argument("--target_update_interval", type=int, default=1, metavar="",
                        help="Value target update per number of updates per step (default: 1)")

    parser.add_argument("--initial_epsilon", type=float, default=0, metavar="",
                        help="Initial value of epsilon, which is the probability of "
                             "random sampling environment actions (default: 0) ")

    parser.add_argument("--epsilon_decay", type=float, default=0, metavar="",
                        help="Exponential decay of epsilon (default: 0)"
                             "[meaningless without --initial_epsilon] "
                             "[conflicts with --epsilon_decrease]")

    parser.add_argument("--epsilon_decrease", type=float, default=0, metavar="",
                        help="Linear decrease of epsilon (default: 0)"
                             "[meaningless without --initial_epsilon] "
                             "[conflicts with --epsilon_decay]")

    parser.add_argument("--final_epsilon", type=float, default=0, metavar="",
                        help="Final value of epsilon (default: 0) "
                             "[meaningless with --initial_epsilon and --epsilon_decrease]")

    parser.add_argument("--learning_starts", type=int, default=0, metavar="",
                        help="How many steps of the model to collect transitions for "
                             "before learning starts (default: 0)")

    parser.add_argument("--max_episode_steps", type=int, default=None, metavar="",
                        help="Maximum number of timesteps per episode (default: None "
                             "[Environment dependent])")

    parser.add_argument("--exploratory_steps", type=int, default=0, metavar="",
                        help="Number of exploratory (i.e. with random actions) "
                             "initial steps (default: None)")

    parser.add_argument("--seed", type=int, default=0, metavar="",
                        help="Random seed (default: 0)")

    parser.add_argument("--render", action="store_true",
                        help="Render simulation (default: False)")

    parser.add_argument("--verbose", type=int, default=1, metavar="",
                        help="Verbose level [0..3] (default: 1)")

    parser.add_argument("--load_params", type=str, default=None, metavar="",
                        help="Directory with the neural networks parameters to be loaded (default: None)")

    parser.add_argument("--save_params", action="store_true",
                        help="Save the neural networks parameters [need --train]")

    parser.add_argument("--save_params_dir", type=str, default=None, metavar="",
                        help="Directory to which to save the neural networks parameters "
                             "at the end of the training (default: None)")

    parser.add_argument("--train", action="store_true",
                        help="Perform training")

    parser.add_argument("--train_episodes", type=int, default=None, metavar="",
                        help="Maximum number of episodes (default: None)")

    parser.add_argument("--test", action="store_true",
                        help="Perform a test [after training, in case]")

    parser.add_argument("--test_episodes", type=int, default=None, metavar="",
                        help="Number of testing episodes (default None -> limitless) "
                             "[meaningless without --test]")

    parser.add_argument("--plot", action="store_true",
                        help="Plot reward curve")

    parser.add_argument("--plot_interval", type=int, default=1, metavar="",
                        help="Number of episodes between plots (default: 1)")

    parser.add_argument("--grid_search", action="store_true",
                        help="Perform a grid search on hyperparameters instead of usual training")

    return parser


def smooth(data, smoothness=0.25):
    smoothed = list()
    n = len(data)
    k = int(smoothness * len(data) / 2)
    for i, data_point in enumerate(data):
        a = max(0, int(i - k))
        b = min(n, int(i + k))
        smoothed_val = np.mean(data[a:b+1])
        smoothed.append(smoothed_val)

    return smoothed

def plot_data(data, title, x_label, y_label):
    if not mlp.is_interactive():
        plt.ion()
    plt.clf()
    smoothed_data = smooth(data)
    plt.plot(np.arange(1, len(data)+1), smoothed_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)

def plot_episodes_return(episodes_return):
    title = "Return per episode"
    x_label = "Episode"
    y_label = "Return"
    plot_data(episodes_return, title, x_label, y_label)

def save_plot():
    plt.ioff()
    dir = "Plots/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    fname = dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_plot"
    print("Saving plot to {}.".format(fname))
    plt.savefig(fname)

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


def params_grid_search(env, args, n_calls=100, verbose=True):
    global ENV, ARGS
    ENV = env
    ARGS = args

    params_range_list = [
        np.linspace(0.90, 0.99),    # gamma
        np.logspace(-4, -1, 10),    # initial_alpha
        np.logspace(-4, -1, 10)     # lr
    ]

    res = skopt.gp_minimize(func=cumulated_loss,
                            dimensions=params_range_list,
                            n_calls=n_calls,
                            verbose=verbose)

    print(res.x, res.fun)
    joblib.dump(res, open('res.pkl', 'wb'))


def cumulated_loss(params):
    global ENV, ARGS
    gamma, initial_alpha, lr = params
    ARGS.gamma = gamma
    ARGS.initial_alpha = initial_alpha
    ARGS.lr = lr
    ARGS.max_episodes = 200

    agent = SAC(ENV.observation_space, ENV.action_space, ARGS)

    cumulated_return = train(ENV, agent, ARGS, return_type=1)
    ENV.close()
    return -cumulated_return