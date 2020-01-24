import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_data(data, title, x_label, y_label):
    sns.set()
    plt.plot(range(1, len(data)+2), data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def plot_return(episode_returns):
    episode_returns = np.array(episode_returns)
    title = 'Return per episode. Min: {:.2f}. Max: {:.2f}. Avg: {:.2f}'.format(
        episode_returns.min(),
        episode_returns.max(),
        episode_returns.mean()
    )

    plot_data(episode_returns, title, 'Episode', 'Return')


def plot_steps(episode_steps):
    episode_steps = np.array(episode_steps)
    title = 'Steps per episode. Min: {:d}. Max: {:d}. Avg: {:.2f}'.format(
        episode_steps.min(),
        episode_steps.max(),
        episode_steps.mean()
    )

    plot_data(episode_steps, title, 'Episode', 'Steps')
