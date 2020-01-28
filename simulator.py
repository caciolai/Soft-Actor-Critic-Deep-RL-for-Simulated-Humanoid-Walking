import itertools
import numpy as np

from replay_buffer import ReplayBuffer
from utils import plot_mean_k_episodes_return


def train(env, agent, args, return_type=0):

    if args.max_episode_steps is not None:
        env.set_max_episode_steps(args.max_episode_steps)

    replay_buffer = ReplayBuffer(args.replay_size)

    total_steps = 0
    updates = 0
    returns = []
    epsilon = args.initial_epsilon

    for i_episode in itertools.count(1):
        state = env.reset()
        episode_return = 0
        i_step = 0

        for i_step in itertools.count(0):
            if args.render:
                env.render()

            if total_steps <= args.exploratory_steps:
                action = env.action_space.sample()
            elif epsilon > 0 and np.random.uniform(0,1) <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)
            if args.verbose >= 2:
                print("Step: {}".format(i_step))
                print("(s,a,r,s',d): ({}, {}, {}, {}, {})".format(state, action, reward, next_state, done))

            replay_buffer.append(state, action, reward, next_state, done)

            if total_steps > args.learning_starts and len(replay_buffer) > args.batch_size:
                for _ in range(args.gradient_steps):
                    q1l, q2l, pl, al = agent.update(replay_buffer, args.batch_size, updates)
                    if args.verbose >= 2:
                        print("Losses: ({}, {}, {}, {})".format(q1l, q2l, pl, al))
                    updates += 1

            state = next_state
            episode_return += reward
            i_step += 1
            total_steps += 1

            if done:
                break

        returns.append(episode_return)
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

        if args.plot and i_episode % args.plot_interval == 0:
            plot_mean_k_episodes_return(returns)

        if args.train_episodes is not None and i_episode >= args.train_episodes:
            break

        if epsilon > 0 and \
        total_steps > args.learning_starts and \
        total_steps > args.exploratory_steps:
            if args.epsilon_decrease > 0 and epsilon > args.final_epsilon:
                epsilon = max(args.final_epsilon, epsilon - args.epsilon_decrease)
            elif args.epsilon_decay > 0:
                epsilon *= args.epsilon_decay

    return np.array(returns)



def test(env, agent, test_episodes):
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

        if test_episodes is not None and \
        i_episode >= test_episodes:
            break
    env.close()

