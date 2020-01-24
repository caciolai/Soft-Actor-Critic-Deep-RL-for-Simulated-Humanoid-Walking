import datetime
import random
from tensorboardX import SummaryWriter
from replay_buffer import ReplayBuffer


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

    i_episode = 1
    total_steps = 0
    updates = 0
    epsilon = args.initial_epsilon

    episodes_return_list = []
    episodes_steps_list = []

    # action_magnitudes = []
    try:
        while i_episode < args.max_steps:
            episode_return = 0
            episode_steps = 0
            done = False
            state = env.reset()

            while not done:
                if args.render:
                    env.render()

                # # sample action from epsilon random policy
                if epsilon is not None and random.uniform(0,1) <= epsilon:
                    action = env.action_space.sample()
                elif args.exploratory_steps is not None and total_steps < args.exploratory_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state)

                # action_magnitudes.append(np.abs(action))

                # perform action and observe next state and reward
                next_state, reward, done, _ = env.step(action)

                if args.custom_reward:
                    reward = custom_reward(next_state, reward)

                if args.verbose >= 2:
                    print(next_state, reward, done)

                # Append transition to replay buffer
                replay_buffer.append(state, action, reward, next_state, float(done))
                state = next_state

                # if replay buffer is sufficiently large to contain at least a minibatch
                if len(replay_buffer) > args.minibatch_size:
                    for i in range(args.gradient_steps):
                        # update parameters of all the networks
                        value_loss, q1_loss, q2_loss, policy_loss = agent.update_networks_parameters(
                            replay_buffer,
                            args.minibatch_size,
                            updates
                        )

                        if args.verbose >= 2:
                            print("Value loss: {:.3f}. Q1 loss: {:.3f}. Q2 loss: {:.3f}. Policy loss: {:.3f}".format(
                                q1_loss, q2_loss, value_loss, policy_loss
                            ))

                        # write losses to tensorboard for visualization
                        if args.tensorboard:
                            writer.add_scalar("loss/Q1", q1_loss, updates)
                            writer.add_scalar("loss/Q2", q2_loss, updates)
                            writer.add_scalar("loss/value", value_loss, updates)
                            writer.add_scalar("loss/policy", policy_loss, updates)

                        updates += 1

                episode_return += reward
                episode_steps += 1
                total_steps += 1

                # if max number of episode steps has been exceeded
                if episode_steps > args.max_episode_steps:
                    break

            # print/write stats to tensorboard
            if args.tensorboard:
                writer.add_scalar("episode_return", episode_return, i_episode)
                writer.add_scalar("episode_steps", episode_steps, i_episode)
                writer.add_scalar("epsilon_randomness", epsilon, i_episode)
                # writer.add_scalar("mean action magnitude", np.array(action_magnitudes).mean(), i_episode)

            if args.verbose >= 1:
                print("Episode: {}, "
                      "total steps: {}, "
                      "episode steps: {}, "
                      "episode return: {:.3f}, "
                      "epsilon randomness: {:.3f}".format(i_episode,
                                                        total_steps,
                                                        episode_steps,
                                                        episode_return,
                                                        epsilon))

            if args.save_params_interval and i_episode % args.save_params_interval == 0:
                agent.save_networks_parameters(prefix)

            episodes_return_list.append(episode_return)
            episodes_steps_list.append(episode_steps)

            # if total number of steps has been exceeded
            if total_steps >= args.max_steps:
                break

            i_episode += 1

            # # epsilon decay
            # if epsilon is not None:
            #     epsilon *= args.epsilon_decay

            # epsilon linear decrease
            if epsilon > args.final_epsilon:
                epsilon -= args.epsilon_decrease

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("\nTerminating...")
        if args.save_params_interval:
            agent.save_networks_parameters(prefix)
        else:
            agent.save_networks_parameters()
        env.close()

        return episodes_return_list, episodes_steps_list


def rescale(value, old_min, old_max, new_min, new_max):
    # Figure out how 'wide' each range is
    old_range = old_max - old_min
    new_range = new_max - new_min

    # Convert the left range into a 0-1 range (float)
    normalized_value = float(value - old_min) / float(old_range)

    # Convert the 0-1 range into a value in the right range.
    return new_min + (normalized_value * new_range)

def custom_reward(state, reward, max_custom_reward=1):
    distance = 0.6 - state[0]

    distance = rescale(distance, 0, 1.6, 0, 1)
    custom_term = max_custom_reward * ((1-distance)**2)

    return reward + custom_term