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
class BasePolicyNetwork(nn.Module):
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, action_range=1, 
            log_std_min=-20, log_std_max=2, hidden_size=256, init_w=3e-3, 
            name='policy', chkpt_dir='tmp/', method='sac'):
        super(BasePolicyNetwork, self).__init__()

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
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        pass
    
    def sample_normal(self, state):
        pass

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def sample_action(self):
        action = T.FloatTensor(self.action_dim).uniform_(-1, 1)
        return (self.action_range * action).numpy()