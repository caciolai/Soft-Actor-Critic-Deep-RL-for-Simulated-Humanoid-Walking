import gym
import torch
import numpy as np
from texttable import Texttable
import traceback
from warnings import simplefilter

from simulator import train, test
from sac import SAC
from utils import FeaturizedStates, NormalizedActions, build_argsparser, save_plot

params_dir = None

def main():
    simplefilter(action="ignore", category=UserWarning)
    simplefilter(action="ignore", category=FutureWarning)
    parser = build_argsparser()
    args = parser.parse_args()

    # environment setup

    env = NormalizedActions(gym.make("MountainCarContinuous-v0"))
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
                global params_dir
                params_dir = agent.save_networks_parameters(args.save_params_dir)

            if args.plot:
                save_plot()

            env.close()

    # testing
    if args.test:
        try:
            # env = FeaturizedStates(NormalizedActions(gym.make("MountainCarContinuous-v0")))
            env = NormalizedActions(gym.make("MountainCarContinuous-v0"))
            agent = SAC(env.observation_space, env.action_space, args)

            if params_dir is None:
                if args.load_params is None:
                    print("WARNING: Testing a random agent.")
                else:
                    params_dir = args.load_params
                    print("Using selected parameters.")
            else:
                print("Using training parameters.")

            agent.load_networks_parameters(params_dir)

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

