import time
import gym
import numpy as np
import random
import torch
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from src.commons import get_modified_state, set_seed_everywhere, WandbLogger, get_yaml_args, get_remaining_time
from src.td3 import Agent


if __name__ == '__main__':
    method = 'td3_mod_state'
    mode = 'train'

    conf = get_yaml_args(method, mode)

    # Seeding
    seed = conf.seed
    no_graphics = conf.no_graphics
    environment_name = conf.environment_name
    fast_forward = conf.fast_forward
    environment_folder = conf.environment_folder

    deterministic = conf.deterministic

    # Which states to remove
    to_remove = conf.to_remove

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
    actor_lr = conf.policy_lr

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
        side_channels=[channel], no_graphics=True, seed=seed)
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

    best_score = env.reward_range[0]
    score_history = []

    start_time = time.time()
    timestep = 0

    try:
        for episode in range(episodes):
            observation = env.reset()
            observation = get_modified_state(observation, to_remove=to_remove)
            done = False
            score = 0

            epoch_time = time.time()
            policy_losses, q1_losses, q2_losses = [], [], []
            while not done:
                timestep += 1
                action = agent.choose_action(
                    observation,
                    deterministic=deterministic,
                    explore_noise_scale=explore_noise_scale
                )

                observation_, reward, done, info = env.step(action)
                observation_ = get_modified_state(observation_, to_remove=to_remove)

                score += reward
                agent.remember(observation, action, reward, observation_, done)
                if agent.memory.mem_cntr > agent.batch_size:
                    q1_, q2_, p_ = agent.learn(
                        debug=False,
                        deterministic=deterministic,
                        eval_noise_scale=eval_noise_scale
                    )

                    # Log losses
                    if p_:
                        policy_losses.append(p_.detach().cpu().numpy())
                    q1_losses.append(q1_.detach().cpu().numpy())
                    q2_losses.append(q2_.detach().cpu().numpy())

                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if conf.wandb_log:
                LOGGER.plot_metrics({
                        'cumulative_reward': score,
                        'episode': episode,
                        'timestep': timestep
                    })
                LOGGER.plot_metrics({
                        'avg_reward': avg_score,
                        'episode': episode,
                        'timestep': timestep
                    })
                LOGGER.plot_epoch_loss(
                    'policy_epoch_loss_ave', policy_losses, episode, timestep)
                LOGGER.plot_epoch_loss(
                    'q1_epoch_loss_ave', q1_losses, episode, timestep)
                LOGGER.plot_epoch_loss(
                    'q2_epoch_loss_ave', q2_losses, episode, timestep)

            elapsed_time = time.time() - start_time
            remaining_time = get_remaining_time(episodes, start_time, episode)

            if episode > 0 and episode % conf.save_frequency == 0 and avg_score > best_score:
                best_score = avg_score
                agent.save_models()
            
            print(f"################## Episode: {episode} ##################")
            print("Elapsed time: ", elapsed_time)
            print("Episode time: ", time.time() - epoch_time)
            print('Score %.1f' % score, 'avg_score %.1f' % avg_score)
            print(f'Remaining_time: {remaining_time}s')
    except KeyboardInterrupt:
        print("Exited!")
    # Safely close the environment
    env.close()
