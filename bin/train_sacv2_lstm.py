import gym
import numpy as np
import random
import torch
from src.sac_v2_lstm import Agent
from src.commons import plot_learning_curve, NormalizedActions, set_seed_everywhere, WandbLogger
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import argparse
import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", help="wandb project",
                        type=str, action='store', nargs='?')
    parser.add_argument("--name", help="wandb experiment name",
                        type=str, action='store', nargs='?')
    parser.add_argument("--entity", help="wandb entity name", 
                        type=str, action='store', nargs='?')
    parser.add_argument("--epochs", help="epochs", default=250,
                        type=int)
    parser.add_argument("--seed", help="seed", 
                        type=int, default=42)
    parser.add_argument("--save", help="save frequency", 
                        type=int, default=50)
    parser.add_argument("--ff", help="fast forward simulation", 
                        type=int, default=20)
    parser.add_argument("--wandb")

    args = parser.parse_args()
    SEED = args.seed
    set_seed_everywhere(SEED)

    # Environment
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale = args.ff)
    environment_name = 'RL_Simulator_paper_nographics'
    unity_env = UnityEnvironment(
        './unity_environments/RL_Simulator_paper_nographics/RL_Simulator_paper_nographics', 
        side_channels=[channel], no_graphics=True, seed=SEED)
    env = UnityToGymWrapper(unity_env)

    action_dim = env.action_space.shape[0]
    state_dim  = 50

    # Method
    method = 'sac_v2_lstm'
    
    # hyperparameters
    n_games = args.epochs
    batch_size = 512
    hidden_dim = 128
    auto_entropy = True
    deterministic = False
    reward_scale = 10.

    agent = Agent(state_dim=state_dim, env=env, batch_size = batch_size, hidden_dim=hidden_dim,
            action_dim=action_dim, action_range=1, reward_scale=reward_scale)

    filename = f'{method}_{environment_name}.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    # Logger
    if args.wandb:
        LOGGER = WandbLogger(
                    project=args.proj,
                    name=args.name,
                    entity=args.entity
                )

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    start_time = time.time()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        # Replay Buffer containers
        last_action = env.action_space.sample()
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []

        hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to('cpu'), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).to('cpu'))

        # Logging variables
        epoch_time = time.time()
        policy_losses, q1_losses, q2_losses = [], [], []

        step = 0
        while not done:
            hidden_in = hidden_out
            action, hidden_out = agent.choose_action(observation, last_action, hidden_in, deterministic=deterministic)
            observation_, reward, done, info = env.step(action)
            
            score += reward

            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out
            episode_state.append(observation)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(observation_)
            episode_done.append(done)

            observation = observation_
            last_action = action

            if not load_checkpoint and len(agent.memory.buffer) > agent.batch_size:
                q1_, q2_, p_ = agent.learn(
                    auto_entropy=auto_entropy, 
                    target_entropy=-1.*action_dim,
                    debug=False
                )

                # Log losses
                policy_losses.append(p_.detach().cpu().numpy())
                q1_losses.append(q1_.detach().cpu().numpy())
                q2_losses.append(q2_.detach().cpu().numpy())

            step += 1
        agent.remember(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if args.wandb:
            LOGGER.plot_metrics('avg_reward', avg_score)
            LOGGER.plot_metrics('reward', score_history),
            LOGGER.plot_epoch_loss('policy_epoch_loss_ave', policy_losses)
            LOGGER.plot_epoch_loss('q1_epoch_loss_ave', q1_losses)
            LOGGER.plot_epoch_loss('q2_epoch_loss_ave', q2_losses)

        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = (elapsed_time / (i + 1)) * (n_games - (i + 1))

        if i > 0 and i % args.save == 0 and avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print("Elapsed time: ", elapsed_time)
        print("Epoch time: ", time.time() - epoch_time)
        print('Episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        print(f'Remaining_time: {remaining_time}s')

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
    
    # Safely close the environment
    env.close()
