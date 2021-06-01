import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SoftQNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, action_dim, name,
                checkpoint_dir='tmp/ddpg'):
        super(SoftQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dim = action_dim
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg')

        # Initialize Network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

        # Initialize layer with weights and biases with smaller variance
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2  = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.f2_dims)

        self.action_value = nn.Linear(self.action_dim, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.load_state_dict(torch.load(self.checkpoint_file))