from src.td3_lstm.policy_network import PolicyNetworkLSTM
from src.td3_lstm.q_network import QNetworkLSTM
from src.commons import ReplayBufferLSTM2 as ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np


class Agent():
    def __init__(self, alpha=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2,
            hidden_size=128, batch_size=512, reward_scale=1, policy_target_update_interval=1,
            device='cpu'):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.policy_net = PolicyNetworkLSTM(alpha, state_dim, action_dim=action_dim,
                    name='policy_net', action_range=action_range, method='td3_lstm',
                    hidden_size=hidden_size, device=device)
        self.target_policy_net = PolicyNetworkLSTM(alpha, state_dim, action_dim=action_dim,
                    name='target_policy_net', action_range=action_range, method='td3_lstm',
                    hidden_size=hidden_size, device=device)
        self.q_net1 = QNetworkLSTM(alpha, state_dim, action_dim=action_dim,
                    name='q_net1', method='td3_lstm', hidden_size=hidden_size, device=device)
        self.q_net2 = QNetworkLSTM(alpha, state_dim, action_dim=action_dim,
                    name='q_net2', method='td3_lstm', hidden_size=hidden_size, device=device)
        self.target_q_net1 = QNetworkLSTM(alpha, state_dim, action_dim=action_dim,
                    name='target_q_net1', method='td3_lstm', hidden_size=hidden_size, device=device)
        self.target_q_net2 = QNetworkLSTM(alpha, state_dim, action_dim=action_dim,
                    name='target_q_net2', method='td3_lstm', hidden_size=hidden_size, device=device)

        self.reward_scale = reward_scale
        self.update_network_parameters(1)

        self.action_range = action_range
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

    def choose_action(self, state, last_action, hidden_in, explore_noise_scale=0.5):
        return self.policy_net.choose_action(
                state, last_action, hidden_in, explore_noise_scale=explore_noise_scale)

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
        self.target_policy_net.load_checkpoint()
        self.target_q_net1.load_checkpoint()
        self.target_q_net2.load_checkpoint()
        self.q_net1.load_checkpoint()
        self.q_net2.load_checkpoint()

    def learn(self, eval_noise_scale=1.0, deterministic=True, debug=False):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done \
            = self.memory.sample(self.batch_size)

        state      = T.FloatTensor(state).to(self.policy_net.device)
        next_state = T.FloatTensor(next_state).to(self.policy_net.device)
        action     = T.FloatTensor(action).to(self.policy_net.device)
        last_action = T.FloatTensor(last_action).to(self.policy_net.device)
        reward     = T.FloatTensor(reward).unsqueeze(-1).to(self.policy_net.device)  
        done       = T.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.policy_net.device)

        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        new_action,  _= self.policy_net.evaluate(state, last_action, hidden_in, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out, eval_noise_scale=eval_noise_scale) # clipped normal noise
        
        reward = self.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = T.min(predicted_target_q1, predicted_target_q2)

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
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

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