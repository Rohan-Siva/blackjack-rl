import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions, state_shape, hidden_sizes=[64, 64]):
        super(DQN, self).__init__()

        layers = []
        input_size = state_shape[0] if isinstance(state_shape, list) else state_shape
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, num_actions))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
