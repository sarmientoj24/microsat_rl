import torch.nn.functional as F 
import torch as T
import torch as T
from src.sac.soft_q_network import SoftQNetwork
import torch.nn as nn

class SoftQNetworkV2(SoftQNetwork):
    def __init__(self, alpha=0.0001,  state_dim=50, action_dim=4, hidden_size=128, 
            init_w=3e-3, name='softq', chkpt_dir='./tmp/', method='sacv2'):
        super(SoftQNetworkV2, self).__init__(
            alpha=alpha,  state_dim=state_dim, action_dim=action_dim, 
            hidden_size=hidden_size, init_w=init_w, 
            name=name, chkpt_dir=chkpt_dir, method=method)
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.q(x)
        return x