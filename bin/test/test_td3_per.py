import time
import gym
import numpy as np
import random
import torch
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from src.commons import NormalizedActions, set_seed_everywhere, WandbLogger, get_yaml_args, get_remaining_time
from src.td3_per import Agent
import pandas as pd


if __name__ == '__main__':
    method = 'td3_per'
    mode = 'test'

    conf = get_yaml_args(method, mode)

    # Seeding
    seed = conf.seed
    no_graphics = conf.no_graphics
    environment_name = conf.environment_name
    fast_forward = conf.fast_forward
    environment_folder = conf.environment_folder

    deterministic = conf.deterministic

    action_dim = conf.action_dim
    state_dim  = conf.state_dim

    # Hyper parameters
    episodes = conf.episodes
    batch_size = conf.batch_size
    hidden_size = conf.hidden_size
    reward_scale = conf.reward_scale
    action_range = conf.action_range
    explore_noise_scale = conf.explore_noise_scale
    policy_target_update_interval = conf.policy_target_update_interval
    update_itr = conf.update_itr
    eval_noise_scale = conf.eval_noise_scale
    critic_lr = conf.critic_lr
    policy_lr = conf.policy_lr

    # Device
    if conf.device == 'cpu':
        device = conf.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # wandb
    if conf.wandb_log:
        LOGGER = WandbLogger(
                    project=conf.proj,
                    name=conf.name,
                    entity=conf.entity
                )

    # Set Seed
    set_seed_everywhere(conf.seed)

    # Configure Environment
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale = fast_forward)
    unity_env = UnityEnvironment(
        f'./{environment_folder}/{environment_name}/{environment_name}', 
        side_channels=[channel], no_graphics=no_graphics, seed=seed)
    env = UnityToGymWrapper(unity_env)

    # Agent
    agent = Agent(
        state_dim=state_dim, 
        batch_size=batch_size, 
        hidden_size=hidden_size,
        action_dim=action_dim, 
        action_range=action_range, 
        reward_scale=reward_scale,
        device=device,
        policy_target_update_interval=policy_target_update_interval,
        method=method,
        policy_lr=policy_lr,
        critic_lr=critic_lr
    )
    
    if conf.pretrain:
        agent.load_models()

    try:
        for episode in range(2):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(
                    observation,
                    deterministic=deterministic,
                    explore_noise_scale=explore_noise_scale,
                    test=True
                )

                observation_, reward, done, info = env.step(action)
                score += reward
                observation = observation_
            print(f"########## Episode {episode}")
            print(f"\tCumulative score: {score}")
    except KeyboardInterrupt:
        print("Exited!")
    # Safely close the environment
    env.close()
