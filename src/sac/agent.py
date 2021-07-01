from src.sac.value_network import ValueNetwork
from src.sac.q_network import QNetwork
from src.sac.policy_network import PolicyNetwork
from src.commons import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np

class Agent():
    def __init__(self, policy_lr=0.0001, critic_lr=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2,
            hidden_size=128, batch_size=512, reward_scale=1, device='cpu', method='sac', alpha_term=1.0):
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
        self.value_net = ValueNetwork(alpha=critic_lr, state_dim=state_dim, name='value',
                    hidden_size=hidden_size,  method=method, device=device)
        self.target_value_net = ValueNetwork(alpha=critic_lr, state_dim=state_dim, name='target_value',
                    hidden_size=hidden_size,  method=method, device=device)

        self.reward_scale = reward_scale
        self.update_network_parameters(1)

        self.action_range = action_range

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

        target_value_params = self.target_value_net.named_parameters()
        value_params = self.value_net.named_parameters()
        self.target_value_net.load_state_dict(
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
        self.value_net.save_checkpoint()
        self.target_value_net.save_checkpoint()
        self.q_net1.save_checkpoint()
        self.q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        self.value_net.load_checkpoint()
        self.target_value_net.load_checkpoint()
        self.q_net1.load_checkpoint()
        self.q_net2.load_checkpoint()

    def learn(self, debug=False):
        alpha = 1.0

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).unsqueeze(1).to(self.policy_net.device)
        done = T.tensor(np.float32(done)).unsqueeze(1).to(self.policy_net.device)
        next_state = T.tensor(new_state, dtype=T.float).to(self.policy_net.device)
        state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
        action = T.tensor(action, dtype=T.float).to(self.policy_net.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        predicted_value    = self.value_net(state)
        new_action, log_prob = self.policy_net.sample_normal(state)

        reward = self.reward_scale*(reward - reward.mean(dim=0)) / ((reward.std(dim=0) + 1e-6)) # normalize with batch mean and std
        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value # if done==1, only reward
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())

        self.q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.q_net1.optimizer.step()
        self.q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.q_net2.optimizer.step()  

        # Training Value Function
        predicted_new_q_value = T.min(self.q_net1(state, new_action), self.q_net2(state, new_action))
        target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action

        value_loss = F.mse_loss(predicted_value, target_value_func.detach())

        self.value_net.optimizer.zero_grad()
        value_loss.backward()
        self.value_net.optimizer.step()

        # Training Policy Function
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()
        
        if debug:
            print('value_loss: ', value_loss)
            print('q loss: ', q_value_loss1, q_value_loss2)
            print('policy loss: ', policy_loss )
        
        self.update_network_parameters()
        return (value_loss, q_value_loss1, q_value_loss2, policy_loss)