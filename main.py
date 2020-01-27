import gym
import numpy as np
import torch
from texttable import Texttable
import traceback
import warnings

from sac import SAC
from utils import *
from train import *

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = build_parser()
    args = parser.parse_args()

    # environment setup
    # env = NormalizedActions(gym.make("Pendulum-v0"))
    env = FeaturizedStates(NormalizedActions(gym.make("Pendulum-v0")))
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

        print("\nUsing device: {}".format(torch.cuda.get_device_properties(agent.device)))
        print("\nStarting training.")

    # training
    try:
        train(env, agent, args)
    except KeyboardInterrupt:
        print("\nInterrupt received.")
    except Exception:
        traceback.print_exc()
    finally:
        print("\nTraining terminated.")
        env.close()

    if args.testing:
        input("\nPress ENTER to initiate testing.")
        # testing
        try:
            env = FeaturizedStates(NormalizedActions(gym.make("Pendulum-v0")))
            test(env, agent, args.testing_steps)
        except KeyboardInterrupt:
            print("\nInterrupt received.")
        except Exception:
            traceback.print_exc()
        finally:
            print("\nTesting terminated.")
            env.close()


if __name__ == "__main__":
    main()

