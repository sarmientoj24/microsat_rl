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
    def __init__(self, alpha, num_inputs, n_actions, max_action=1, hidden_size=256, 
            init_w=3e-3, name='policy', chkpt_dir='tmp/sac'):
        super(PolicyNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, n_actions)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        
        self.sigma = nn.Linear(hidden_size, n_actions)
        self.sigma.weight.data.uniform_(-init_w, init_w)
        self.sigma.bias.data.uniform_(-init_w, init_w)

        # Range of 4 actions are already normalized [-1, 1]
        self.action_range = max_action
        self.n_actions = n_actions
        self.reparam_noise = 1e-6

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        mean    = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mean, sigma
    
    def sample_normal(self, state):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, std = self.forward(state)
        
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
        action = T.FloatTensor(self.n_actions).uniform_(-1, 1)
        return (self.action_range * action).numpy()