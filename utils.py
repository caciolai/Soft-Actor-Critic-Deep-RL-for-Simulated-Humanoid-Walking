import argparse
import datetime
import os
import gym
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

"""
This file contains utility methods needed in other files of the project
"""

def build_argparser():
    """
    Builds the argparser to read arguments from command-line
    :return: the parser with all the arguments set and ready to be read
    """
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
                        help="Verbose level [0..2] (default: 1)")

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

    return parser


def mean_k(data, k):
    """
    Given some data, each data point is replaced by its local k mean
    :param data: data
    :type: list
    :param k: size of the neighborhood to consider for averaging
    :type k: int
    :return: the averaged data
    """
    n = len(data)
    mean_k_data = list()
    for i, datapoint in enumerate(data):
        a = max(0, int(np.ceil(i - k/2)))
        b = min(n, int(np.floor(i + k/2)))
        mean_k_datapoint = np.mean(data[a:b+1])
        mean_k_data.append(mean_k_datapoint)

    return mean_k_data


def plot_data(data, title, x_label, y_label):
    """
    Plots data interactively
    :param data: data to be plot
    :type data: list or numpy array or similar
    :param title: title of the plot figure
    :type title: str
    :param x_label: label for the x-axis
    :type x_label: str
    :param y_label: label for the y-axis
    :type y_label: str
    :return: None
    """
    if not mlp.is_interactive():                    # activate pyplot interactive mode
        plt.ion()
    plt.clf()                                       # clear currently drawn figure
    plt.plot(np.arange(1, len(data)+1), data)       # plot data (x-axis is [1..len(data)])
    plt.title(title)                                # set title
    plt.xlabel(x_label)                             # set xlabel
    plt.ylabel(y_label)                             # set ylabel
    plt.grid(True)                                  # show grid for ease of visualization
    plt.show()                                      # show figure
    plt.pause(0.001)                                # small interval to actually see something in between "frames"


def plot_mean_k_episodes_return(episodes_return, frac=0.1):
    """
    Plots the mean k episode return obtained in the simulation
    :param episodes_return: array containing the cumulative reward earned in each episode so far
    :type episodes_return: list or numpy array
    :param frac: parameter for smoothing, by default 0.1 meaning each data point gets replaced by
                    the mean over its neighborhood of size 1/10 of the whole data (so it can be considered
                    a smoothing coefficient)
    :type frac: float [0,1]
    :return: None
    """
    sns.set()
    k = int(np.ceil(frac * len(episodes_return)))
    k = min(k, 100)
    title = "Mean {} episode return".format(k)
    x_label = "Episode"
    y_label = "Return"
    plot_data(mean_k(episodes_return, k), title, x_label, y_label)

def save_plot():
    """
    Saves the currently active figure, in the Plots directory in the current directory,
    with name based on the current timestamp
    :return: None
    """
    plt.ioff()
    plots_dir = "Plots/"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fname = os.path.join(plots_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_plot")
    print("Saving plot to {}.".format(fname))
    plt.savefig(fname)

# action in [-1, 1] to action in [low, high]
class NormalizedActions(gym.ActionWrapper):
    """
    A class that inherits from gym's ActionWrapper, since it is meant to translate actions ``normalized''
    in [-1,1] to actions in the correct environment action space [low, high]
    """
    def __init__(self, env):
        """
        Constructor that simply calls super class constructor
        :param env: The environment being wrapped
        :type env: OpenAI gym environment
        """
        super().__init__(env)

    def action(self, action):
        """
        Takes an action in [-1, 1] and translates it to [low, high]
        :param action: action
        :type action: numpy array of dtype float32
        :return: translated action
        """
        # inherited from gym.ActionWrapper which took it from env
        low = self.action_space.low
        high = self.action_space.high

        # if action is -1, then high terms cancel out and it remains low, the reverse if action is +1
        action = ((high + low) + action*(high - low)) / 2.0

        # should not be needed, but in case actions were in fact outside of [-1, 1] this corrects the above
        # calculation
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        """
        Takes an action in [low, high] and translates it back to [-1, 1], thus reversing the above operations
        Not sure if ever gets executed, but it actually needs to be implemented
        since it is an abstract method of superclass
        :param action: action
        :type action: numpy array of dtype float32
        :return: translated action
        """
        low = self.action_space.low
        high = self.action_space.high

        action = (2.0 * action - (high + low)) / (high - low)
        action = np.clip(action, -1, 1)

        return action

    def get_max_episode_steps(self):
        """
        Getter to access the protected max_episode_steps of the environment
        :return: maximum number of timesteps per episode before the environment sends a positive done signal
        :rtype: int
        """
        return self.env._max_episode_steps

    def set_max_episode_steps(self, num):
        """
        Setter to override the time limit of the environment,
        often needed to ensure enough time for initial exploration
        :param num: desired maximum number of timesteps per episode
        :return: None
        """
        self.env._max_episode_steps = num

