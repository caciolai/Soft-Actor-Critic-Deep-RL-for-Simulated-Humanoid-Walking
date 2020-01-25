import gym
import numpy as np
import torch
from sac import SAC
from texttable import Texttable

from utils import *
from train import *

def main():
    parser = build_parser()
    args = parser.parse_args()

    # environment setup
    env = gym.envs.make("Pendulum-v0")
    env._max_episode_steps = args.max_episode_steps
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # agent
    agent = SAC(env.observation_space, env.action_space, args)
    params_path = args.load_params
    agent.load_networks_parameters(params_path)

    if args.verbose >= 1:
        print("Setup completed. Settings:\n")
        t = Texttable()
        t.set_cols_dtype(['t', 'e'])
        t.add_rows([["Argument", "Value"]] + [[arg, getattr(args, arg)] for arg in vars(args)])
        print(t.draw())

        print("\nEpisode time horizon: {} steps.".format(env._max_episode_steps))
        print("\nObservation space shape: {}".format(env.observation_space.shape))
        print("Observation space range: [{}, {}]".format(
            env.observation_space.low, env.observation_space.high)
        )
        print("\nAction space shape: {}".format(env.action_space.shape))
        print("Action space range: [{}, {}]".format(env.action_space.low, env.action_space.high))
        print("\nUsing device: {}".format(torch.cuda.get_device_properties(agent.device)))
        print("\nStarting training.")

    # training
    train(env, agent, args)

    # testing
    test(env, agent, 1e4)


if __name__ == "__main__":
    main()

