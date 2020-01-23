import argparse

def handle_parser():
    parser = argparse.ArgumentParser(description="PyTorch SAC")

    parser.add_argument("--render", action="store_true",
                        help="Render simulation (default: False)")

    parser.add_argument("--tensorboard", action="store_true",
                        help="Use tensorboard for tracking (default: False)")

    parser.add_argument("--verbose", type=int, default=1, metavar="",
                        help="Verbose level [1..3] (default: 1)")

    parser.add_argument("--alpha", type=float, default=0.2, metavar="",
                        help="Temperature parameter α determines the relative importance of "
                             "the entropy term against the reward (default: 0.2)")

    parser.add_argument("--gamma", type=float, default=0.99, metavar="",
                        help="Discount factor for reward (default: 0.99)")

    parser.add_argument("--tau", type=float, default=0.005, metavar="",
                        help="Target smoothing coefficient (default: 0.005)")

    parser.add_argument("--lr", type=float, default=0.0003, metavar="",
                        help="Learning rate (default: 3e-4)")

    parser.add_argument("--hidden_units", type=int, default=256, metavar="",
                        help="Number of units in hidden layers (default: 256)")

    parser.add_argument("--replay_size", type=int, default=1000000, metavar="",
                        help="Size of replay buffer (default: 1e6)")

    parser.add_argument("--seed", type=int, default=0, metavar="",
                        help="Random seed (default: 0)")

    parser.add_argument("--minibatch_size", type=int, default=256, metavar="",
                        help="Minibatch size (default: 256)")

    parser.add_argument("--epsilon", type=float, default=0.95, metavar="",
                        help="Exponential factor of decrease of epsilon randomness (default: 0.95)")

    parser.add_argument("--max_steps", type=int, default=1000000, metavar="",
                        help="Maximum number of timesteps (default: 1e6)")

    parser.add_argument("--max_episode_steps", type=int, default=1000, metavar="",
                        help="Maximum number of timesteps per episode (default: 1000)")

    parser.add_argument("--exploratory_steps", type=int, default=10000, metavar="",
                        help="Number of exploratory (i.e. with random actions) "
                             "initial steps (default: 1e4)")

    parser.add_argument("--gradient_steps", type=int, default=1, metavar="",
                        help="Gradient steps per simulator step (default: 1)")

    parser.add_argument("--target_update_interval", type=int, default=1, metavar="",
                        help="Value target update per number of updates per step (default: 1)")

    parser.add_argument("--save_params_interval", type=int, default=None, metavar="",
                        help="If set, interval of episodes to save net params (default: None)")

    parser.add_argument("--load_policy", type=str, default=None, metavar="",
                        help="Dir of Policy network parameters (default: None)")

    parser.add_argument("--load_q_function", type=str, default=None, metavar="",
                        help="Dir of (both) Q function network parameters (default: None)")

    parser.add_argument("--load_value_function", type=str, default=None, metavar="",
                        help="Dir of Value function network parameters (default: None)")

    return parser


