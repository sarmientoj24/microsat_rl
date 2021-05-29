import gym
import numpy as np
import random
import torch
import pybullet_envs
from src.sac import Agent
from src.commons import plot_learning_curve, NormalizedActions, set_seed_everywhere, WandbLogger
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", help="wandb project", required=True,
                        type=str)
    parser.add_argument("--name", help="wandb experiment name", required=True,
                        type=str)
    parser.add_argument("--entity", help="wandb entity name",
                        type=str)
    parser.add_argument("--epochs", help="epochs", default=250,
                        type=int)
    parser.add_argument("--seed", help="seed", 
                        type=int, default=42)

    args = parser.parse_args()
    SEED = args.seed
    set_seed_everywhere(SEED)

    # Environment
    environment_name = 'RL_Simulatorv2'
    unity_env = UnityEnvironment('./unity_environments/RL_Simulatorv2/RL_Simulatorv2')
    env = UnityToGymWrapper(unity_env)

    action_dim = env.action_space.shape[0]
    state_dim  = 50

    agent = Agent(num_inputs=state_dim, env=env,
            n_actions=action_dim, max_action=1)
    n_games = args.epochs

    filename = f'{environment_name}.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False


    # Logger
    LOGGER = WandbLogger(
                project=args.proj,
                name=args.name,
                entity=args.entity
            )

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        policy_losses, value_losses, q1_losses, q2_losses = [], [], [], []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint and agent.memory.mem_cntr > agent.batch_size:
                v_, q1_, q2_, p_ = agent.learn(debug=False)

                # Log losses
                policy_losses.append(p_.detach().cpu().numpy())
                value_losses.append(v_.detach().cpu().numpy())
                q1_losses.append(q1_.detach().cpu().numpy())
                q2_losses.append(q2_.detach().cpu().numpy())

            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        LOGGER.plot_metrics('avg_reward', avg_score)
        LOGGER.plot_metrics('reward', score_history),
        LOGGER.plot_epoch_loss('policy_epoch_loss_ave', policy_losses)
        LOGGER.plot_epoch_loss('value_epoch_loss_ave', value_losses)
        LOGGER.plot_epoch_loss('q1_epoch_loss_ave', q1_losses)
        LOGGER.plot_epoch_loss('q2_epoch_loss_ave', q2_losses)

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)