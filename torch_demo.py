import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import json

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return np.argmax(act_values.detach().numpy())  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy())
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.model.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Manifest:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
        self.video_time = self.data["Video_Time"]
        self.chunk_count = self.data["Chunk_Count"]
        self.chunk_time = self.data["Chunk_Time"]
        self.buffer_size = self.data["Buffer_Size"]
        self.available_bitrates = self.data["Available_Bitrates"]
        self.preferred_bitrate = self.data["Preferred_Bitrate"]
        self.chunks = [self.data["Chunks"][str(i)] for i in range(self.chunk_count)]

class Simulator:
    def __init__(self, manifest):
        self.available_bitrates = manifest.available_bitrates  # Load from manifest
        self.total_video_length = manifest.video_time  # Load from manifest
        self.chunk_time = manifest.chunk_time  # Load chunk time from manifest
        self.chunks = manifest.chunks  # Load chunk data from manifest
        self.reset()

    def reset(self):
        # Initialize variables for a new episode
        self.measured_bandwidth = 1000  # Starting value (in Kbps)
        self.previous_throughput = 800  # Starting value (in Kbps)
        self.buffer_occupancy = 0  # Buffer starts empty (in seconds)
        self.video_time = 0  # Video time starts at 0 seconds
        self.rebuffering_time = 0  # No rebuffering at the start
        self.preferred_bitrate = self.available_bitrates[0]  # Set preferred bitrate to the first available bitrate

        initial_state = np.array([
            self.measured_bandwidth,
            self.previous_throughput,
            self.buffer_occupancy,
            self.preferred_bitrate,
            self.video_time,
            self.rebuffering_time
        ])

        return initial_state

    def step(self, action):
        selected_bitrate = self.available_bitrates[action]  # Get selected bitrate (in Kbps)
        chunk_size = (selected_bitrate * 1000) * self.chunk_time / 8  # Convert to bytes

        # Simulate downloading the chunk based on the selected bitrate
        if self.measured_bandwidth < selected_bitrate:
            # Simulate rebuffering if the measured bandwidth is less than the selected bitrate
            self.rebuffering_time += 1  # Increment rebuffering time
            reward = -1  # Penalty for rebuffering
            self.buffer_occupancy = max(self.buffer_occupancy - 1, 0)  # Decrease buffer occupancy
        else:
            # Simulate successful download
            self.buffer_occupancy += self.chunk_time  # Add the chunk duration to buffer
            self.video_time += self.chunk_time  # Increment the video time
            reward = 1  # Reward for successful playback

            # Update previous throughput
            self.previous_throughput = self.measured_bandwidth

        # Official simulator score calculation
        

        # Update measured bandwidth (simulate some variability)
        self.measured_bandwidth = np.random.randint(800, 1200)  # Simulate bandwidth variation (Kbps)

        # Determine if the video has finished
        done = self.video_time >= self.total_video_length

        # Create the next state
        next_state = np.array([
            self.measured_bandwidth,
            self.previous_throughput,
            self.buffer_occupancy,
            self.preferred_bitrate,
            self.video_time,
            self.rebuffering_time
        ])

        return next_state, reward, done

def train_agent(episodes, save_path="dqn_model.pth"):
    state_size = 8  # Change this to the number of state variables you have
    action_size = 5  # Change this to the number of available bitrates
    agent = DQNAgent(state_size, action_size)
    manifest = Manifest("manifest.json")
    simulator = Simulator(manifest)

    for e in range(episodes):
        state = simulator.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = simulator.step(action)  # Get the next state and reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(32)  # Replay experience and update the model
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Save the model every certain number of episodes
        if (e + 1) % 100 == 0:  # Save every 100 episodes
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_agent(1000)  # Train for 1000 episodes