from src.td3.policy_network import PolicyNetwork
from src.td3.q_network import QNetwork
from src.commons import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np


class Agent():
    def __init__(self, policy_lr=0.0001, critic_lr=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2,
            hidden_size=128, batch_size=512, reward_scale=1, policy_target_update_interval=1,
            device='cpu', method='td3'):
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer(max_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.policy_net = PolicyNetwork(alpha=policy_lr, state_dim=state_dim, action_dim=action_dim,
                    name='policy_net', action_range=action_range, method=method,
                    hidden_size=hidden_size, device=device)
        self.target_policy_net = PolicyNetwork(alpha=policy_lr, state_dim=state_dim, action_dim=action_dim,
                    name='target_policy_net', action_range=action_range, method=method,
                    hidden_size=hidden_size, device=device)
        self.q_net1 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='q_net1', method=method, hidden_size=hidden_size, device=device)
        self.q_net2 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='q_net2', method=method, hidden_size=hidden_size, device=device)
        self.target_q_net1 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='target_q_net1', method=method, hidden_size=hidden_size, device=device)
        self.target_q_net2 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='target_q_net2', method=method, hidden_size=hidden_size, device=device)

        self.reward_scale = reward_scale
        self.update_network_parameters(1)

        self.action_range = action_range
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

    def choose_action(self, state, deterministic=True, explore_noise_scale=0.5, test=False):
        if test:
            return self.policy_net.choose_action(
                state, deterministic=deterministic, explore_noise_scale=explore_noise_scale)
        if self.memory.mem_cntr < self.batch_size:
            return self.policy_net.sample_action()
        else:
            return self.policy_net.choose_action(
                state, deterministic=deterministic, explore_noise_scale=explore_noise_scale)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_params = self.target_q_net1.named_parameters()
        base_params = self.q_net1.named_parameters()
        self.target_q_net1.load_state_dict(
            self.network_update(base_params, target_params, tau)
        )

        target_params = self.target_q_net2.named_parameters()
        base_params = self.q_net2.named_parameters()
        self.target_q_net2.load_state_dict(
            self.network_update(base_params, target_params, tau)
        )

        target_params = self.target_policy_net.named_parameters()
        base_params = self.policy_net.named_parameters()
        self.target_policy_net.load_state_dict(
            self.network_update(base_params, target_params, tau)
        )

    def network_update(self, base_params, target_params, tau):
        target_dict = dict(target_params)
        state_dict = dict(base_params)

        for name in state_dict:
            state_dict[name] = tau*state_dict[name].clone() + \
                    (1-tau)*target_dict[name].clone()
        return state_dict

    def save_models(self):
        print('.... saving models ....')
        self.policy_net.save_checkpoint()
        self.target_policy_net.save_checkpoint()
        self.target_q_net1.save_checkpoint()
        self.target_q_net2.save_checkpoint()
        self.q_net1.save_checkpoint()
        self.q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        # self.target_policy_net.load_checkpoint()
        self.target_q_net1.load_checkpoint()
        self.target_q_net2.load_checkpoint()
        self.q_net1.load_checkpoint()
        self.q_net2.load_checkpoint()

    def learn(self, eval_noise_scale, deterministic=True, debug=False):
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).unsqueeze(1).to(self.policy_net.device)
        done = T.tensor(np.float32(done)).unsqueeze(1).to(self.policy_net.device)
        next_state = T.tensor(new_state, dtype=T.float).to(self.policy_net.device)
        state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
        action = T.tensor(action, dtype=T.float).to(self.policy_net.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action, log_prob = self.policy_net.sample_normal(state, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _ = self.target_policy_net.sample_normal(next_state, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        reward = self.reward_scale * (reward - reward.mean(dim=0)) / ((reward.std(dim=0) + 1e-6)) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = T.min(self.target_q_net1(next_state, new_next_action),self.target_q_net2(next_state, new_next_action))
        target_q_value = reward + (1 - done) * self.gamma * target_q_min # if done==1, only reward
        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        self.q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.q_net1.optimizer.step()
        self.q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.q_net2.optimizer.step()

        policy_loss = None
        if self.update_cnt % self.policy_target_update_interval == 0:
        # Training Policy Function
            predicted_new_q_value = self.q_net1(state, new_action)
            policy_loss = - predicted_new_q_value.mean()
            self.policy_net.optimizer.zero_grad()
            policy_loss.backward()
            self.policy_net.optimizer.step()
        
            # Soft update the target nets
            self.update_network_parameters()

        self.update_cnt+=1
        
        if debug:
            print('q loss: ', q_value_loss1, q_value_loss2)
        return (q_value_loss1, q_value_loss2, policy_loss)