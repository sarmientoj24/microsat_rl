import torch.nn.functional as F 
import torch as T
import torch as T
from src.sac.soft_q_network import SoftQNetwork
import torch.nn as nn

class SoftQNetworkV2(SoftQNetwork):
    def __init__(self, beta,  num_inputs, n_actions=4, hidden_size=256, init_w=3e-3, 
            name='softq', chkpt_dir='./tmp/', method='sacv2'):
        super(SoftQNetworkV2, self).__init__(
            beta=beta,  num_inputs=num_inputs, n_actions=n_actions, 
            hidden_size=hidden_size, init_w=init_w, 
            name=name, chkpt_dir=chkpt_dir, method=method)
        
        self.fc1 = nn.Linear(num_inputs + n_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        
        self.fc4.weight.data.uniform_(-init_w, init_w)
        self.fc4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = T.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x