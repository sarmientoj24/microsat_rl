import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import copy
from src.commons.networks.network_init import *
import os


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_dim=50, 
            name='value', chkpt_dir='./tmp/', method=''):
        super(ValueNetworkBase, self).__init__()

        self.state_dim = state_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)

    def forward(self):
        pass

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_dim=50, action_dim=4, 
            name='qnet', chkpt_dir='./tmp/', method=''):
        super().__init__(state_dim=state_dim, name=name, method=method)
        self.action_dim = action_dim


class ValueNetwork(ValueNetworkBase):
    def __init__(self, alpha=0.0001, hidden_dim=256, state_dim=50, 
            name='value', chkpt_dir='./tmp/', method=''):
        super().__init__(state_dim=state_dim, name=name, method=method)

        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = self.F.relu(self.linear1(state))
        x = self.F.relu(self.linear2(x))
        x = self.linear3(x)
        return x        


class QNetwork(QNetworkBase):
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, hidden_dim=256, 
            name='qnet', chkpt_dir='./tmp/', method=''):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name=name, method=method)

        self.linear1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.apply(linear_weights_init)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        x = T.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x        

class QNetworkLSTM(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, hidden_dim=256, 
            name='qnet_lstm', chkpt_dir='./tmp/', method=''):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name=name, method=method)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self.state_dim+self.action_dim, hidden_dim)
        self.linear2 = nn.Linear(self.state_dim+self.action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = T.cat([state, action], -1) 
        fc_branch = F.relu(self.linear1(fc_branch))
        # branch 2
        lstm_branch = T.cat([state, last_action], -1) 
        lstm_branch = F.relu(self.linear2(lstm_branch))  # linear layer for 3d input only applied on the last dim
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = T.cat([fc_branch, lstm_branch], -1) 

        x = F.relu(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   

class QNetworkLSTM2(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, hidden_dim=256, 
            name='qnet_lstm', chkpt_dir='./tmp/', method=''):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name=name, method=method)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self.state_dim + 2*self.action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = T.cat([state, action, last_action], -1) 
        x = F.relu(self.linear1(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   