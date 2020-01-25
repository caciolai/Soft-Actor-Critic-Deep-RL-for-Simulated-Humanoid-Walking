import datetime
import itertools
import random
import traceback

import numpy as np
from tensorboardX import SummaryWriter
from replay_buffer import ReplayBuffer
from utils import plot


def train(env, agent, args):

    # saved agents dir
    if args.save_params_interval:
        prefix = "SavedAgents/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


    # tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(
            logdir="TensorBoardLogs/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # replay replay_buffer
    replay_buffer = ReplayBuffer(args.replay_size)

    total_steps = 0
    returns = []

    for i_episode in itertools.count(1):
        state = env.reset()
        episode_return = 0
        i_step = 0

        while i_step < args.max_episode_steps:
            if total_steps > args.exploratory_steps:
                action = agent.policy_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)

            replay_buffer.append(state, action, reward, next_state, done)

            if len(replay_buffer) > args.batch_size:
                agent.update(replay_buffer, args.batch_size)


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


def test(env, agent, steps=50000):

    # Run a demo of the environment
    state = env.reset()
    for step in range(steps):
        env.render()
        action = agent.policy_net.get_action(state)
        state, reward, done, info = env.step(action.detach())
        if done:
            break
    env.close()