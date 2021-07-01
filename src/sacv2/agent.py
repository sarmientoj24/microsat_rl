from src.sacv2.q_network import QNetwork
from src.sacv2.policy_network import PolicyNetwork
from src.commons import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class Agent():
    def __init__(self, policy_lr=0.0001, critic_lr=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2, alpha_lr=3e-4,
            hidden_size=128, batch_size=512, reward_scale=1, device='cpu', method='sac'):
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer(max_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.policy_net = PolicyNetwork(alpha=policy_lr, state_dim=state_dim, action_dim=action_dim,
                    name='policy_net', action_range=action_range, method=method,
                    hidden_size=hidden_size, device=device)
        self.q_net1 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='q_net1', hidden_size=hidden_size,  method=method, device=device)
        self.q_net2 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='q_net2', hidden_size=hidden_size,  method=method, device=device)
        self.target_q_net1 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='target_q_net1', hidden_size=hidden_size,  method=method, device=device)
        self.target_q_net2 = QNetwork(alpha=critic_lr, state_dim=state_dim, action_dim=action_dim,
                    name='target_q_net2', hidden_size=hidden_size,  method=method, device=device)

        self.reward_scale = reward_scale
        self.update_network_parameters(1)

        self.action_range = action_range
        self.log_alpha = T.zeros(1, dtype=T.float32, requires_grad=True, device=device)
        self.alpha_lr = alpha_lr
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def choose_action(self, state, deterministic=False, test=False):
        if test:
            return self.policy_net.choose_action(state, deterministic=deterministic)
        if self.memory.mem_cntr < self.batch_size:
            return self.policy_net.sample_action()
        else:
            return self.policy_net.choose_action(state, deterministic=deterministic)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_q_net1.named_parameters()
        value_params = self.q_net1.named_parameters()
        self.target_q_net1.load_state_dict(
            self.network_update(value_params, target_value_params, tau)
        )

        target_value_params = self.target_q_net2.named_parameters()
        value_params = self.q_net2.named_parameters()
        self.target_q_net2.load_state_dict(
            self.network_update(value_params, target_value_params, tau)
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
        self.target_q_net1.save_checkpoint()
        self.target_q_net2.save_checkpoint()
        self.q_net1.save_checkpoint()
        self.q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        self.target_q_net1.load_checkpoint()
        self.target_q_net2.load_checkpoint()
        self.q_net1.load_checkpoint()
        self.q_net2.load_checkpoint()

    def learn(self, debug=False, auto_entropy=True, target_entropy=-2):
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).unsqueeze(1).to(self.policy_net.device)
        done = T.tensor(np.float32(done)).unsqueeze(1).to(self.policy_net.device)
        next_state = T.tensor(new_state, dtype=T.float).to(self.policy_net.device)
        state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
        action = T.tensor(action, dtype=T.float).to(self.policy_net.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action, log_prob = self.policy_net.sample_normal(state)
        new_next_action, next_log_prob = self.policy_net.sample_normal(next_state)

        reward = self.reward_scale*(reward - reward.mean(dim=0)) / ((reward.std(dim=0) + 1e-6)) # normalize with batch mean and std
        
        # Update alpha wrt entropy
        if auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        target_q_min = T.min(self.target_q_net1(next_state, new_next_action), 
                            self.target_q_net2(next_state, new_next_action)) - \
                            self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min

        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())
        
        self.q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.q_net1.optimizer.step()

        self.q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.q_net2.optimizer.step()

        predicted_new_q_value = T.min(self.q_net1(state, new_action), self.q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()

        if debug:
            print('q loss: ', q_value_loss1, q_value_loss2)
            print('policy loss: ', policy_loss)
            print('alpha loss: ', alpha_loss)
        
        self.update_network_parameters()
        return (alpha_loss, q_value_loss1, q_value_loss2, policy_loss)