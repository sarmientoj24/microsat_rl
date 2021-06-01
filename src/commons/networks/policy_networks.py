import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import math
from src.commons.networks.network_init import *
import os


class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_dim=50, action_dim=4, action_range=1, 
            name='policy', chkpt_dir='tmp/', method=''):
        super(PolicyNetworkBase, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range

        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self):
        a = T.FloatTensor(self.action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class SAC_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, action_range=1, 
            log_std_min=-20, log_std_max=2, 
            name='policy', chkpt_dir='tmp/', method='', init_w=3e-3, param_noise=1e-6):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name=name, method=method)
        
        self.linear1 = nn.Linear(self.state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, self.action_dim)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        
        self.log_std = nn.Linear(hidden_size, self.action_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.reparam_noise = param_noise
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean    = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_normal(self, state):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = T.tanh(mean + std * z.to(self.device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - T.log(
            1. - action_0.pow(2) + self.reparam_noise) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def choose_action(self, state, deterministic=True):
        state = T.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.forward(state)
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(self.device)
        action = self.action_range * T.tanh(mean + std * z)

        action = self.action_range * T.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
            action.detach().cpu().numpy()[0]
        return action


class SAC_PolicyNetworkLSTM(PolicyNetworkBase):
    def __init__(self, alpha=0.0001, state_dim=50, hidden_size=128, action_dim=4, action_range=1,
            log_std_min=-20, log_std_max=2, 
            name='policy', chkpt_dir='tmp/', method='', init_w=3e-3, param_noise=1e-6):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name=name, method=method)

        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self.state_dim, hidden_size)
        self.linear2 = nn.Linear(self.state_dim+self.action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, self.action_dim)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        
        self.log_std = nn.Linear(hidden_size, self.action_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.reparam_noise = param_noise
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = F.relu(self.linear1(state))
        # branch 2
        lstm_branch = T.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=T.cat([fc_branch, lstm_branch], -1) 
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, lstm_hidden
    
    def sample_normal(self, state, last_action, hidden_in):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = T.tanh(mean + std * z.to(self.device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - T.log(
            1. - action_0.pow(2) + self.reparam_noise) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def choose_action(self, state, last_action, hidden_in, deterministic=True):
        state = T.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # increase 2 dims to match with training data
        last_action = T.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.device)
        mean, std, hidden_out = self.forward(state, last_action, hidden_in)
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(self.device)
        action = self.action_range * T.tanh(mean + std * z)

        action = self.action_range * T.tanh(mean).detach().cpu().numpy() if deterministic else \
            action.detach().cpu().numpy()
        return action[0][0], hidden_out