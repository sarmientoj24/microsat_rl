from src.sac import PolicyNetwork
from src.sac_v2 import SoftQNetworkV2 as SoftQNetwork
from src.commons import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, num_inputs=4,
            env=None, gamma=0.99, n_actions=4, max_action=1, max_size=1000000, tau=1e-2,
            hidden_dim=256, batch_size=256, reward_scale=1):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.memory = ReplayBuffer(max_size, num_inputs, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.policy_net = PolicyNetwork(alpha, num_inputs, n_actions=n_actions,
                    name='policy_net', max_action=max_action, method='sacv2')
        self.soft_q_net1 = SoftQNetwork(beta, num_inputs, n_actions=n_actions,
                    name='soft_q_net1', method='sacv2')
        self.soft_q_net2 = SoftQNetwork(beta, num_inputs, n_actions=n_actions,
                    name='soft_q_net2', method='sacv2')
        self.target_soft_q_net1 = SoftQNetwork(beta, num_inputs, n_actions=n_actions,
                    name='target_soft_q_net2', method='sacv2')
        self.target_soft_q_net2 = SoftQNetwork(beta, num_inputs, n_actions=n_actions,
                    name='target_soft_q_net2', method='sacv2')

        self.reward_scale = reward_scale
        self.action_range = max_action
        self.log_alpha = T.zeros(1, dtype=T.float32, requires_grad=True, device=self.policy_net.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)
        self.update_network_parameters(tau=1)


    def choose_action(self, state, deterministic=False):
        if self.memory.mem_cntr < self.batch_size:
            return self.policy_net.sample_action()
        else:
            return self.policy_net.choose_action(state)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
        #     target_param.data.copy_(  # copy data value into target parameters
        #         target_param.data * (1.0 - tau) + param.data * tau
        #     )

        # for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
        #     target_param.data.copy_(  # copy data value into target parameters
        #         target_param.data * (1.0 - tau) + param.data * tau
        #     )

        # Soft Q1
        target_soft_q_params = self.target_soft_q_net1.named_parameters()
        soft_q_params = self.soft_q_net1.named_parameters()

        target_soft_q_state_dict = dict(target_soft_q_params)
        soft_q_state_dict = dict(soft_q_params)

        for name in soft_q_state_dict:
            soft_q_state_dict[name] = tau*soft_q_state_dict[name].clone() + \
                    (1-tau)*target_soft_q_state_dict[name].clone()

        self.target_soft_q_net1.load_state_dict(soft_q_state_dict)

        # Soft Q2
        target_soft_q2_params = self.target_soft_q_net2.named_parameters()
        soft_q2_params = self.soft_q_net2.named_parameters()

        target_soft_q2_state_dict = dict(target_soft_q2_params)
        soft_q2_state_dict = dict(soft_q2_params)

        for name in soft_q2_state_dict:
            soft_q2_state_dict[name] = tau*soft_q2_state_dict[name].clone() + \
                    (1-tau)*target_soft_q2_state_dict[name].clone()

        self.target_soft_q_net2.load_state_dict(soft_q2_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.policy_net.save_checkpoint()
        self.soft_q_net1.save_checkpoint()
        self.soft_q_net2.save_checkpoint()
        self.target_soft_q_net1.save_checkpoint()
        self.target_soft_q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        self.soft_q_net1.load_checkpoint()
        self.soft_q_net2.load_checkpoint()
        self.target_soft_q_net1.load_checkpoint()
        self.target_soft_q_net2.load_checkpoint()

    def learn(self, debug=False, auto_entropy=True, target_entropy=-2):
        alpha = 1.0

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).unsqueeze(1).to(self.policy_net.device)
        done = T.tensor(np.float32(done)).unsqueeze(1).to(self.policy_net.device)
        next_state = T.tensor(new_state, dtype=T.float).to(self.policy_net.device)
        state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
        action = T.tensor(action, dtype=T.float).to(self.policy_net.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob = self.policy_net.sample_normal(state)
        new_next_action, next_log_prob = self.policy_net.sample_normal(next_state)

        reward = self.reward_scale*(reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std
        
        if auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.apha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0
        
        # Training Q Function
        target_q_min = T.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min # if done==1, only reward
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())

        self.soft_q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.soft_q_net1.optimizer.step()
        self.soft_q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.soft_q_net2.optimizer.step()  

        # Training Policy Function
        predicted_new_q_value = T.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha* log_prob - predicted_new_q_value).mean()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()
        

        if debug:
            print('q value loss1: ', q_value_loss1)
            print('q value loss2: ', q_value_loss2)
            print('policy loss: ', policy_loss )

        self.update_network_parameters()

        return (q_value_loss1, q_value_loss2, policy_loss)