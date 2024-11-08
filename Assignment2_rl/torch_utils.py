import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import random
import rl_simulator
import time

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.fc7(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)  # Target network
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_model.to(self.device)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)  # Add batch dimension
        act_values = self.model(state)
        return np.argmax(act_values.detach().cpu().numpy())  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)  # Add batch dimension
            next_state = torch.FloatTensor(next_state).to(self.device)  # Add batch dimension
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model(next_state).detach().cpu().numpy())
            target_f = self.model(state)
            target_f[action] = target
            self.model.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            self.loss = loss
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay