from src.sac import ValueNetwork, SoftQNetwork, PolicyNetwork
from src.commons import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np

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
                    name='policy_net', max_action=max_action)
        self.soft_q_net1 = SoftQNetwork(beta, num_inputs, n_actions=n_actions,
                    name='soft_q_net1')
        self.soft_q_net2 = SoftQNetwork(beta, num_inputs, n_actions=n_actions,
                    name='soft_q_net2')
        self.value_net = ValueNetwork(beta, num_inputs, name='value')
        self.target_value_net = ValueNetwork(beta, num_inputs, name='target_value')

        self.reward_scale = reward_scale
        self.update_network_parameters(tau=1)
        self.action_range = max_action

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

        target_value_params = self.target_value_net.named_parameters()
        value_params = self.value_net.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value_net.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.policy_net.save_checkpoint()
        self.value_net.save_checkpoint()
        self.target_value_net.save_checkpoint()
        self.soft_q_net1.save_checkpoint()
        self.soft_q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        self.value_net.load_checkpoint()
        self.target_value_net.load_checkpoint()
        self.soft_q_net1.load_checkpoint()
        self.soft_q_net2.load_checkpoint()

    def learn(self, debug=False):
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
        predicted_value    = self.value_net(state)
        new_action, log_prob = self.policy_net.sample_normal(state)

        reward = self.reward_scale*(reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std
        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value # if done==1, only reward
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())

        self.soft_q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.soft_q_net1.optimizer.step()
        self.soft_q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.soft_q_net2.optimizer.step()  

        # Training Value Function
        predicted_new_q_value = T.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
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