import gym
import torch
import numpy as np
from texttable import Texttable
import traceback
from warnings import simplefilter

from simulator import train, test
from sac import SAC
from utils import FeaturizedStates, NormalizedActions, build_argsparser, save_plot


PARAMS_DIR  = None
# ENV_NAME    = "MountainCarContinuous-v0"
ENV_NAME    = "Humanoid-v2"


def main():
    simplefilter(action="ignore")
    parser = build_argsparser()
    args = parser.parse_args()

    # environment setup

    env = NormalizedActions(gym.make(ENV_NAME))
    # env = FeaturizedStates(NormalizedActions(gym.make("MountainCarContinuous-v0")))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # agent
    agent = SAC(env.observation_space, env.action_space, args)
    agent.load_networks_parameters(args.load_params)

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
            print("\nInterrupt received.")
        except Exception:
            traceback.print_exc()
        finally:
            print("\nTraining terminated.")
            if args.save_params or args.test:
                global PARAMS_DIR
                PARAMS_DIR = agent.save_networks_parameters(args.save_params_dir)

            if args.plot:
                save_plot()

            env.close()

    # testing
    if args.test:
        try:
            # env = FeaturizedStates(NormalizedActions(gym.make("MountainCarContinuous-v0")))
            env = NormalizedActions(gym.make(ENV_NAME))
            agent = SAC(env.observation_space, env.action_space, args)

            if PARAMS_DIR is None:
                if args.load_params is None:
                    print("WARNING: Testing a random agent.")
                else:
                    PARAMS_DIR = args.load_params
                    print("Using selected parameters.")
            else:
                print("Using training parameters.")

            agent.load_networks_parameters(PARAMS_DIR)

            input("\nPress any key to begin testing.")
            test(env, agent, args.test_episodes)
        except KeyboardInterrupt:
            print("\nInterrupt received.")
        except Exception:
            traceback.print_exc()
        finally:
            print("\nTesting terminated.")
            env.close()



if __name__ == "__main__":
    main()

