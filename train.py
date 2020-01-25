import datetime
import itertools
import numpy as np

from replay_buffer import ReplayBuffer


def train(env, agent, args):

    if args.max_episode_steps:
        env._max_episode_steps = args.max_episode_steps

    # saved agents dir
    if args.save_params_interval:
        prefix = "SavedAgents/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # replay buffer
    replay_buffer = ReplayBuffer(args.replay_size)

    total_steps = 0
    updates = 0
    returns = []
    epsilon = args.initial_epsilon

    for i_episode in itertools.count(1):
        state = env.reset()
        episode_return = 0
        i_step = 0

        while i_step < env._max_episode_steps:
            if args.render:
                env.render()

            if args.exploratory_steps is not None and total_steps <= args.exploratory_steps:
                action = env.action_space.sample()
            elif epsilon is not None and np.random.uniform(0,1) <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.append(state, action, reward, next_state, done)

            if len(replay_buffer) > args.batch_size:
                agent.update(replay_buffer, args.batch_size, updates)
                updates += 1


            state = next_state
            episode_return += reward
            i_step += 1
            total_steps += 1

            if done:
                break

        returns.append(episode_return)
        print("Episode: {}. Steps: {}. Episode steps: {}. Episode return: {}".format(
            i_episode, total_steps, i_step, episode_return
        ))

        if total_steps > args.max_steps:
            break

        if epsilon is not None:
            if args.epsilon_decrease is not None and epsilon > args.final_epsilon:
                epsilon -= args.epsilon_decrease
            elif args.epsilon_decay is not None:
                epsilon *= args.epsilon_decay



def test(env, agent, testing_steps):

    # Run a demo of the environment
    total_steps = 0
    while total_steps < testing_steps:
        state = env.reset()
        for _ in range(env._max_episode_steps):
            env.render()
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            if done:
                break
    env.close()