import gym
import numpy as np
import torch
from sac_agent import Agent
from texttable import Texttable

from utils import handle_parser
from train import train

def main():
    parser = handle_parser()
    args = parser.parse_args()

    # environment setup
    env = gym.make("MountainCarContinuous-v0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # agent
    agent = Agent(env.observation_space.shape[0], env.action_space, args)
    policy_path, q1_path, q2_path, value_path = args.load_policy, args.load_q1_function, args.load_q2_function, args.load_value_function
    agent.load_networks_parameters(policy_path, q1_path, q2_path, value_path)

    if args.verbose >= 1:
        print("Setup completed. Settings:\n")
        t = Texttable()
        t.set_cols_dtype(['t', 'e'])
        t.add_rows([["Argument", "Value"]] + [[arg, getattr(args, arg)] for arg in vars(args)])
        print(t.draw())

        print("\nEnvironment time horizon: {} steps.".format(env._max_episode_steps))
        print("Episode horizon (min{{max_episode_steps, env_horizon}}): "
              "{} steps.".format(min(env._max_episode_steps, args.max_episode_steps)))

        print("\nUsing device: {}".format(torch.cuda.get_device_properties(agent.device)))
        print("\nStarting training.")

    # training
    train(env, agent, args)


if __name__ == "__main__":
    main()

