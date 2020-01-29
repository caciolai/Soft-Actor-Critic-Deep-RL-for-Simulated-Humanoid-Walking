import itertools
import numpy as np

from replay_buffer import ReplayBuffer
from utils import plot_mean_k_episodes_return

"""
This file handles all that regards the simulation of the interaction of the agent with the environment
"""

def train(env, agent, args):
    """
    Trains the given agent in the given environment,
    following the specification in the arguments passed via command-line
    :param env: environment
    :type env: OpenAI gym environment
    :param agent: agent to be trained
    :type agent: SAC
    :param args: the arguments parsed from command-line
    :type args: object returned by argparse library
    :return: array with the returns per episode cumulated by the agent during training
    :rtype: numpy array of dtype float32
    """

    if args.max_episode_steps is not None:
        # if user has specified a maximum number of steps per episode, set it
        env.set_max_episode_steps(args.max_episode_steps)

    # build replay buffer
    replay_buffer = ReplayBuffer(args.replay_size)

    total_steps = 0
    updates = 0
    returns = []
    epsilon = args.initial_epsilon

    # for each episode counting from 1
    for i_episode in itertools.count(1):
        # reset the environment and the episode counters, and get the initial state
        state = env.reset()
        episode_return = 0
        i_step = 0

        # for each step in the episode
        for i_step in itertools.count(0):
            if args.render:
                env.render()

            # if user has specified a number of initial exploratory steps,
            # then just sample a random action from the environment action space
            # if user has specified an epsilon randomness different from zero (and the exploratory steps are over)
            # then just sample a random action from the environment action space
            # otherwise let the agent choose an appropriate action
            if total_steps <= args.exploratory_steps:
                action = env.action_space.sample()
            elif epsilon > 0 and np.random.uniform(0,1) <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)

            # perform the action and observe the resulting next state, reward and done signal
            next_state, reward, done, _ = env.step(action)

            # if very verbose print per step log
            if args.verbose >= 2:
                print("Step: {}".format(i_step))
                print("(s,a,r,s',d): ({}, {}, {}, {}, {})".format(state, action, reward, next_state, done))

            # append observed transition to replay buffer
            replay_buffer.append(state, action, reward, next_state, done)

            # if user has specified a number of steps without having the agent update its networks (and learn),
            # then skip the update
            # if that phase is over, then proceed to update agent's networks
            if total_steps > args.learning_starts and len(replay_buffer) > args.batch_size:
                for _ in range(args.gradient_steps):
                    q1l, q2l, pl, al = agent.update(replay_buffer, args.batch_size, updates)
                    if args.verbose >= 2:
                        print("Losses: ({}, {}, {}, {})".format(q1l, q2l, pl, al))
                    updates += 1

            # update per step variables and cumulate episode return
            state = next_state
            episode_return += reward
            i_step += 1
            total_steps += 1

            # if received done signal from the environment, then terminate the episode
            if done:
                break

        # append the cumulated episode return to the array
        returns.append(episode_return)

        # if verbose print a summary of the training occurred in the last episode
        if args.verbose >= 1:
            summary = "Episode: {}. Steps: {}. Episode steps: {}. Episode return: {:.3f}.\n".format(
                i_episode, total_steps, i_step, episode_return
            )
            if args.learning_starts > total_steps:
                summary += "Learning starts in: {} steps. ".format(args.learning_starts - total_steps)
            if args.exploratory_steps > total_steps:
                summary += "Exploratory steps left: {}. ".format(args.exploratory_steps - total_steps)
            elif epsilon > 0:
                summary += "Epsilon: {:.3f}.".format(epsilon)

            print(summary)

        # if user has specified plotting, then plot the returns cumulated so far
        if args.plot and i_episode % args.plot_interval == 0:
            plot_mean_k_episodes_return(returns)

        # if user has specified a fixed number of training episodes, check if time is up
        if args.train_episodes is not None and i_episode >= args.train_episodes:
            break

        # update epsilon randomness coefficient,
        # if still positive and if exploratory phase is over and learning has started
        # linear decrease update wins over exponential decay update, in case user specified both
        if epsilon > 0 and \
        total_steps > args.learning_starts and \
        total_steps > args.exploratory_steps:
            if args.epsilon_decrease > 0 and epsilon > args.final_epsilon:
                epsilon = max(args.final_epsilon, epsilon - args.epsilon_decrease)
            elif args.epsilon_decay > 0:
                epsilon *= args.epsilon_decay

    return np.array(returns)


def test(env, agent, args, test_episodes=None):
    """
    Tests the given agent in the given environment
    :param env: environment
    :type env: OpenAI gym environment
    :param agent: agent
    :type agent: SAC
    :param args: command-line arguments
    :type args: argparse parsed object
    :param test_episodes: number of test episodes to perform
                            (default: None, so nonstop)
    :return: array with the returns per episode cumulated by the agent during training
    :rtype: numpy array of dtype float32
    """
    returns = []
    total_steps = 0
    for i_episode in itertools.count(1):
        episode_return = 0
        state = env.reset()
        for i_step in itertools.count(0):
            total_steps += 1
            env.render()
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            episode_return += reward
            if done:
                break

        summary = "Episode: {}. Steps: {}. Episode steps: {}. Episode return: {:.3f}.\n".format(
            i_episode, total_steps, i_step, episode_return
        )
        print(summary)
        returns.append(episode_return)

        if args.plot and i_episode % args.plot_interval == 0:
            plot_mean_k_episodes_return(returns)

        if test_episodes is not None and \
        i_episode >= test_episodes:
            break

        return returns
