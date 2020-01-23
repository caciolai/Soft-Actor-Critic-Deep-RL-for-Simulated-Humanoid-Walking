import argparse

def handle_parser():
    parser = argparse.ArgumentParser(description="PyTorch SAC")
    parser.add_argument("--epsilon", type=float, default=0.99, metavar="FLOAT",
                        help="Exponential factor of decrease of epsilon randomness (default: 0.99)")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="FLOAT",
                        help="Discount factor for reward (default: 0.99)")

    parser.add_argument("--tau", type=float, default=0.005, metavar="FLOAT",
                        help="Target smoothing coefficient (default: 0.005)")

    parser.add_argument("--lr", type=float, default=0.0003, metavar="FLOAT",
                        help="Learning rate (default: 3e-4)")

    parser.add_argument("--alpha", type=float, default=0.2, metavar="FLOAT",
                        help="Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)")

    parser.add_argument("--seed", type=int, default=0, metavar="INT",
                        help="Random seed (default: 0)")

    parser.add_argument("--minibatch_size", type=int, default=256, metavar="INT",
                        help="Minibatch size (default: 256)")

    parser.add_argument("--max_episode_steps", type=int, default=1000, metavar="INT",
                        help="Maximum number of timesteps for episode (default: 1e3)")

    parser.add_argument("--max_steps", type=int, default=1000000, metavar="INT",
                        help="Maximum number of timesteps for episode (default: 1e6)")

    parser.add_argument("--hidden_units", type=int, default=256, metavar="INT",
                        help="Number of units in hidden layers (default: 256)")

    parser.add_argument("--gradient_steps", type=int, default=1, metavar="INT",
                        help="Gradient steps per simulator step (default: 1)")

    parser.add_argument("--target_update_interval", type=int, default=1, metavar="INT",
                        help="Value target update per number of updates per step (default: 1)")

    parser.add_argument("--replay_size", type=int, default=1000000, metavar="INT",
                        help="Size of replay buffer (default: 1e6)")

    parser.add_argument("--render", action="store_true",
                        help="Render simulation (default: False)")

    parser.add_argument("--verbose", type=int, default=1, metavar="INT",
                        help="Verbose level [1..3] (default: 1)")

    parser.add_argument("--tensorboard", action="store_true",
                        help="Use tensorboard for tracking (default: False)")

    parser.add_argument("--load_policy", type=str, default=None, metavar="models/foo",
                        help="Path of Policy network parameters (default: None)")

    parser.add_argument("--load_q1_function", type=str, default=None, metavar="models/foo",
                        help="Path of Q1 function network parameters (default: None)")

    parser.add_argument("--load_q2_function", type=str, default=None, metavar="models/foo",
                        help="Path of Q2 function network parameters (default: None)")

    parser.add_argument("--load_value_function", type=str, default=None, metavar="models/foo",
                        help="Path of Value function network parameters (default: None)")

    parser.add_argument("--save_params_interval", type=int, default=None, metavar="INT",
                        help="If set, interval of episodes to save net params (default: None)")

    return parser


