import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import random

import rl_simulator
import task_gen
import time

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9999
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

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

def train_agent_test(epoches, save_path="dqn_model.pth"):
    
    state_size = 10  # Change this to the number of state variables you have
    action_size = 3  # Change this to the number of available bitrates
    agent = DQNAgent(state_size, action_size)

    for ep in range(epoches):
        
        start_time = time.time()

        # Generate training trace and manifest files
        trace_path = task_gen.gen_trace()
        manifest_path = task_gen.gen_manifest()
        
        rl_simulator.init(trace_path, manifest_path)
        done = False
        total_reward = 0
        score = 0

        while not done:
        # for state in states:
            # action = agent.act(state)

            action, reward, next_state, state, score, done = rl_simulator.loop(agent, score)
            # print(action, reward, next_state, state, done)
            # next_state, reward, done = simulator.step(action)  # Get the next state and reward
            agent.remember(state, action, reward, next_state, done)
            # state = next_state
            total_reward += reward
            # print("Reward: ", reward, "Time:, ", state[5])

        agent.replay(64)  # Replay experience and update the model

        if (ep + 1) % 100 == 0:
            print(f"Episode: {ep+1}/{epoches}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Time: {time.time() - start_time:.2f}")

                # Save the model every certain number of episodes
                # if (e + 1) % 100 == 0:  # Save every 100 episodes
                # print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
            # print(f"Task: {task}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            torch.save({
                'epoch': ep,
                'model_state_dict': agent.model.state_dict(),  # Save model weights
                'optimizer_state_dict': agent.optimizer.state_dict(),  # Save optimizer state
                'loss': agent.loss,  # Save the current loss
            }, save_path)
            print(f"Model saved to {save_path}")


def train_agent(epoches, episodes, save_path="dqn_model.pth"):
    

    tasks = ["badtest", "testALThard", "testALTsoft", "testHD"]
    tasks += ["testHDmanPQtrace", "testPQ"]

    state_size = 10  # Change this to the number of state variables you have
    action_size = 3  # Change this to the number of available bitrates
    agent = DQNAgent(state_size, action_size)

    for ep in range(epoches):
        tasks = random.sample(tasks, len(tasks))
        for task in tasks:
            start_time = time.time()

            trace_path = f"./tests/{task}/trace.txt"
            manifest_path = f"./tests/{task}/manifest.json"

            for e in range(episodes):
                rl_simulator.init(trace_path, manifest_path)
                done = False
                total_reward = 0
                score = 0

                while not done:
                # for state in states:
                    # action = agent.act(state)

                    action, reward, next_state, state, score, done = rl_simulator.loop(agent, score)
                    # print(action, reward, next_state, state, done)
                    # next_state, reward, done = simulator.step(action)  # Get the next state and reward
                    agent.remember(state, action, reward, next_state, done)
                    # state = next_state
                    total_reward += reward
                    # print("Reward: ", reward, "Time:, ", state[5])

                agent.replay(64)  # Replay experience and update the model
                print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Time: {time.time() - start_time:.2f}")

                # Save the model every certain number of episodes
                # if (e + 1) % 100 == 0:  # Save every 100 episodes
                # print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
            print(f"Task: {task}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            torch.save({
                'epoch': e,
                'model_state_dict': agent.model.state_dict(),  # Save model weights
                'optimizer_state_dict': agent.optimizer.state_dict(),  # Save optimizer state
                'loss': agent.loss,  # Save the current loss
            }, save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    total_time_start = time.time()
    # train_agent(100, 100)  # Train 100 round, train 100 episodes in different tests each round
    train_agent_test(10000)  # Train 10000 round
    print(f"Total training time: {time.time() - total_time_start:.2f} seconds")