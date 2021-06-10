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

class PolicyNetworkLSTM(nn.Module):
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, action_range=1, 
            log_std_min=-20, log_std_max=2, hidden_size=128, init_w=3e-3, 
            name='policy', chkpt_dir='./tmp/', method='td3_lstm', device='cpu'):
        super(PolicyNetworkLSTM, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(state_dim + action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_dim) # output dim = dim of action

        self.fc4.weight.data.uniform_(-init_w, init_w)
        self.fc4.bias.data.uniform_(-init_w, init_w)

        # Range of 4 actions are already normalized [-1, 1]
        self.action_range = action_range
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
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
        fc_branch = F.relu(self.fc1(state)) 
        # branch 2
        lstm_branch = T.cat([state, last_action], -1)
        lstm_branch = F.relu(self.fc2(lstm_branch))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        lstm_branch,  lstm_hidden = self.lstm1(lstm_branch, hidden_in)    # no activation after lstm
        # merged
        merged_branch = T.cat([fc_branch, lstm_branch], -1)   
        x = F.relu(self.fc3(merged_branch))
        x = T.tanh(self.fc4(x))
        x = x.permute(1,0,2)  # permute back

        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)
    
    def evaluate(self, state, last_action, hidden_in, eval_noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = eval_noise_scale * normal.sample(action.shape).to(self.device)
        action = self.action_range * action+noise
        return action, hidden_out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def choose_action(self, state, last_action, hidden_in,  explore_noise_scale=1.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = T.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device) # increase 2 dims to match with training data
        last_action = T.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.device)
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = explore_noise_scale * normal.sample(action.shape).to(self.device)
        action = self.action_range * action + noise
        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        normal = Normal(0, 1)
        random_action=self.action_range * normal.sample( (self.action_dim,) )
        return random_action.cpu().numpy()