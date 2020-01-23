import datetime
import numpy as np

def train(env, agent, replay_buffer, writer, args):

    # saved agents dir
    if args.save_params_interval:
        prefix = "SavedAgents/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    i_episode = 1
    total_steps = 0
    updates = 0
    # epsilon = 1
    # action_magnitudes = []
    try:
        while i_episode < args.max_steps:
            episode_return = 0
            episode_steps = 0
            done = False
            state = env.reset()

            # # decreasing the epsilon randomness at each step
            # epsilon *= args.epsilon

            while not done:
                if args.render:
                    env.render()

                # # sample action from epsilon random policy
                # if torch.rand(1)[0] <= epsilon:
                #     action = env.action_space.sample()
                if total_steps < args.exploratory_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state)

                # action_magnitudes.append(np.abs(action))

                # perform action and observe next state and reward
                next_state, reward, done, _ = env.step(action)

                if args.verbose >= 2:
                    print(next_state, reward, done)

                # ignore done signal if not actually dependent on state
                mask = False if episode_steps == env._max_episode_steps else done

                # Append transition to replay buffer
                replay_buffer.append(state, action, reward, next_state, float(mask))
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
                # writer.add_scalar("mean action magnitude", np.array(action_magnitudes).mean(), i_episode)

            if args.verbose >= 1:
                print("Episode: {}, "
                      "total steps: {}, "
                      "episode steps: {}, "
                      "episode return: {:.3f}".format(i_episode,
                                                      total_steps,
                                                      episode_steps,
                                                      episode_return))

            if args.save_params_interval and i_episode % args.save_params_interval == 0:
                agent.save_networks_parameters(prefix)

            # if total number of steps has been exceeded
            if total_steps >= args.max_steps:
                break

            i_episode += 1

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("\nTerminating...")
        if args.save_params_interval:
            agent.save_networks_parameters(prefix)
        else:
            agent.save_networks_parameters()
        env.close()