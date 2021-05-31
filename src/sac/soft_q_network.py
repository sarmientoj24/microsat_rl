import torch.nn.functional as F 
import torch as T
import torch.nn as nn
import os
import torch.optim as optim


class SoftQNetwork(nn.Module):
    def __init__(self, beta,  num_inputs, n_actions=4, hidden_size=256, init_w=3e-3, 
            name='critic', chkpt_dir='./tmp/', method='sac'):
        super(SoftQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs + n_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))