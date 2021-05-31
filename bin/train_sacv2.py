import gym
import numpy as np
import random
import torch
from src.sac_v2 import Agent
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
        # no_graphics=True,
        # base_port=5100,
        # worker_id=1)
    env = UnityToGymWrapper(unity_env)

    action_dim = env.action_space.shape[0]
    state_dim  = 50

    # Method
    method = 'sac_v2'

    # Hyper parameters
    n_games = args.epochs
    batch_size = 512
    hidden_dim = 128
    deterministic = False
    reward_scale = 1.
    auto_entropy = True

    agent = Agent(state_dim=state_dim, env=env, batch_size=512, hidden_dim=128,
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

        epoch_time = time.time()
        policy_losses, q1_losses, q2_losses = [], [], []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint and agent.memory.mem_cntr > agent.batch_size:
                q1_, q2_, p_ = agent.learn(
                    auto_entropy=auto_entropy, 
                    target_entropy=-1.*action_dim,
                    debug=False
                )

                # Log losses
                policy_losses.append(p_.detach().cpu().numpy())
                q1_losses.append(q1_.detach().cpu().numpy())
                q2_losses.append(q2_.detach().cpu().numpy())

            observation = observation_
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
