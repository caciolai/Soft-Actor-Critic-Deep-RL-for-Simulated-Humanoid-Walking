import gym
import torch
import numpy as np
from texttable import Texttable
import traceback
from warnings import simplefilter

from simulator import train, test
from sac import SAC
from utils import NormalizedActions, build_argparser, save_plot


PARAMS_DIR  = None
# ENV_NAME    = "MountainCarContinuous-v0"
# ENV_NAME    = "Pendulum-v0"
ENV_NAME    = "LunarLanderContinuous-v2"
# ENV_NAME    = "Humanoid-v2"


def main():
    """
    The main file of the project
    """

    # args and warnings ignoring setup
    simplefilter(action="ignore")
    parser = build_argparser()
    args = parser.parse_args()

    # environment setup
    env = NormalizedActions(gym.make(ENV_NAME))     # to ensure actions in [-1, 1] get correctly translated
    # setting libraries seeds to try and have repeatability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # agent setup
    agent = SAC(env.observation_space, env.action_space, args)
    agent.load_networks_parameters(args.load_params)

    # if verbose, print a tabular recap of the args passed via command-line (or default ones)
    if args.verbose >= 1:
        t = Texttable()
        t.set_cols_dtype(['t', 'e'])
        t.add_rows([["Argument", "Value"]] +
                   [[arg, getattr(args, arg)] for arg in vars(args)] +
                   [["device", agent.device]])
        print(t.draw())
        print("\nSetup completed. Settings shown in the table above.")

    # training
    if args.train:
        input("\nPress any key to begin training.")
        try:
            train(env, agent, args)
        except KeyboardInterrupt:
            # to stop training
            print("\nInterrupt received.")
        except Exception:
            # if anything else happens, catch the exception and print it but without crashing
            traceback.print_exc()
        finally:
            print("\nTraining terminated.")
            # if required to save parameters, or need them for later testing, save them
            if args.save_params or args.test:
                global PARAMS_DIR
                PARAMS_DIR = agent.save_networks_parameters(args.save_params_dir)

            # save the plot that has been generated so far, if any
            if args.plot:
                save_plot()

            # close the environment
            env.close()

    # testing
    if args.test:
        try:
            # build environment and agent
            env = NormalizedActions(gym.make(ENV_NAME))
            agent = SAC(env.observation_space, env.action_space, args)

            if PARAMS_DIR is None:
                # then look if the user has specified a directory for loading parameters
                if args.load_params is None:
                    # then the agent will not load any parameters and will therefore act purely random
                    print("WARNING: Testing a random agent.")
                else:
                    PARAMS_DIR = args.load_params
                    print("Using selected parameters.")
            else:
                print("Using training parameters.")

            # initialize agent's networks' parameters
            agent.load_networks_parameters(PARAMS_DIR)

            input("\nPress any key to begin testing.")
            test(env, agent, args.test_episodes)
        except KeyboardInterrupt:
            # to stop testing
            print("\nInterrupt received.")
        except Exception:
            # if anything else happens, catch the exception and print it but without crashing
            traceback.print_exc()
        finally:
            print("\nTesting terminated.")
            # save the plot that has been generated so far, if any
            if args.plot:
                save_plot()

            # close the environment
            env.close()


# simply call the main method
if __name__ == "__main__":
    main()

