import torch.nn.functional as F
import torch as T
import torch.nn as nn
import os
import torch.optim as optim


class QNetworkLSTM(nn.Module):
    def __init__(self, alpha=0.0001,  state_dim=50, action_dim=4, hidden_size=128, init_w=3e-3, 
            name='q_net', chkpt_dir='./tmp/', method='td3_lstm', device='cpu'):
        super(QNetworkLSTM, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(state_dim + action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device

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
        fc_branch = F.relu(self.fc1(fc_branch))
        # branch 2
        lstm_branch = T.cat([state, last_action], -1) 
        lstm_branch = F.relu(self.fc2(lstm_branch))  # fc layer for 3d input only applied on the last dim
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = T.cat([fc_branch, lstm_branch], -1) 

        x = F.relu(self.fc3(merged_branch))
        x = self.q(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell) 

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))