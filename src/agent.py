import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from src.model import DQN
from src.utils import ReplayBuffer

class DQNAgent:
    def __init__(self, num_actions, state_shape, hidden_sizes=[64, 64], lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01, buffer_size=20000, batch_size=64, device='cpu'):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(num_actions, state_shape, hidden_sizes).to(device)
        self.target_net = DQN(num_actions, state_shape, hidden_sizes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.criterion = nn.MSELoss()

    def step(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def eval_step(self, state):

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
            expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
