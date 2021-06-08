import torch.nn.functional as F
import torch as T
import torch.nn as nn
import os
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

'''
    This is also called the Actor-Network
'''
class PolicyNetwork(nn.Module):
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, action_range=1, 
            log_std_min=-20, log_std_max=2, hidden_size=128, init_w=3e-3, 
            name='policy', chkpt_dir='tmp/', method='sac', device='cpu'):
        super(PolicyNetwork, self).__init__()
        hidden_size = 256
        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_dim)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

        # Range of 4 actions are already normalized [-1, 1]
        self.action_range = action_range
        self.action_dim = action_dim
        self.reparam_noise = 1e-6
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean    = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_normal(self, state):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = T.tanh(mean + std*z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        ''' stochastic evaluation '''
        log_prob = Normal(mean, std).log_prob(mean + std*z.to(self.device)) - T.log(1. - action_0.pow(2) + self.reparam_noise) -  np.log(self.action_range)
        ''' deterministic evaluation '''
        # log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
        '''
        both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        '''
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def choose_action(self, state, deterministic=False):
        state = T.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.forward(state)
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(self.device)
        action = self.action_range * T.tanh(mean + std*z)        
        action = T.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        
        return action

    def sample_action(self):
        action = T.FloatTensor(self.action_dim).uniform_(-1, 1)
        return (self.action_range * action).numpy()