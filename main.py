import datetime
import gym
import numpy as np
import itertools
import torch
from sac_agent import Agent
from tensorboardX import SummaryWriter
from texttable import Texttable

from replay_buffer import ReplayBuffer
from utils import handle_parser


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

    # tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(
            logdir="TensorBoardLogs/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # saved agents dir
    if args.save_params_interval:
        prefix = "SavedAgents/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # replay buffer
    buffer = ReplayBuffer(args.replay_size)

    if args.verbose >= 1:
        print("Setup completed. Settings:\n")
        t = Texttable()
        t.set_cols_dtype(['t', 'e'])
        t.add_rows([["Argument", "Value"]] + [[arg, getattr(args, arg)] for arg in vars(args)])
        print(t.draw())

        print("\nEnvironment time horizon: {} steps".format(env._max_episode_steps))
        print("Episode horizon (min{{max_episode_steps, env_horizon}}): "
              "{} steps".format(min(env._max_episode_steps, args.max_episode_steps)))

        print("\nUsing device: ", torch.cuda.get_device_properties(agent.device))
        print("\nStarting learning...\n")

    # training
    total_steps = 0
    updates = 0
    try:
        for i_episode in itertools.count(1):
            episode_return = 0
            episode_steps = 0
            done = False
            state = env.reset()

            while not done:
                if args.render:
                    env.render()

                # sample action from policy
                action = agent.select_action(state)

                # perform action and observe state and reward
                next_state, reward, done, _ = env.step(action)
                episode_steps += 1
                total_steps += 1

                if args.verbose >= 2:
                    print(next_state, reward, done)

                # ignore done signal if not actually dependent on state
                mask = False if episode_steps == env._max_episode_steps else done

                episode_return += reward

                # Append transition to replay buffer
                buffer.push(state, action, reward, next_state, float(mask))
                state = next_state

                # if replay buffer is sufficiently large to contain at least a minibatch
                if len(buffer) > args.minibatch_size:
                    for i in range(args.gradient_steps):
                        # update parameters of all the networks
                        value_loss, q1_loss, q2_loss, policy_loss = agent.update_networks_parameters(
                            buffer,
                            args.minibatch_size,
                            updates
                        )

                        if args.verbose >= 2:
                            print("Value loss: {:.3f}. Q1 loss: {:.3f}. Q2 loss: {:.3f}. Policy loss: {:.3f}".format(
                                value_loss, q1_loss, q2_loss, policy_loss
                            ))

                        # write losses to tensorboard for visualization
                        if args.tensorboard:
                            writer.add_scalar("loss/value", value_loss, updates)
                            writer.add_scalar("loss/Q1", q1_loss, updates)
                            writer.add_scalar("loss/Q2", q2_loss, updates)
                            writer.add_scalar("loss/policy", policy_loss, updates)
                        updates += 1


                # if max number of episode steps has been exceeded
                if episode_steps > args.max_episode_steps:
                    break

            # print/write stats to tensorboard
            if args.tensorboard:
                writer.add_scalar("episode_return", episode_return, i_episode)
            if args.verbose >= 1:
                print("Episode: {}, "
                      "total steps: {}, "
                      "episode steps: {}, "
                      "episode return: {}".format(i_episode,
                                                     total_steps,
                                                     episode_steps,
                                                     round(episode_return, 2)))

            if args.save_params_interval and i_episode % args.save_params_interval == 0:
                agent.save_networks_parameters(prefix)

            # if total number of steps has been exceeded
            if total_steps >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("\nTerminating...")
        if args.save_params_interval:
            agent.save_networks_parameters(prefix)
        else:
            agent.save_networks_parameters()
        env.close()


if __name__ == "__main__":
    main()

