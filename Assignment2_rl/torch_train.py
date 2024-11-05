import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import json
from rl_grader import *  # Import the grading functions
import threading

import socket
import studentcode_123090823 as studentcode


# Quick-and-dirty TCP Server:
# ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ss.bind(('localhost', 6000))
ss.listen(10)
# print('Waiting for simulator')

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
        self.loss = 0

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
            self.loss = loss
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
        # #Measured_Bandwidth, Previous_Throughput, Buffer_Occupancy, Available_Bitrates, Video_Time, Chunk, Rebuffering_Time, Preferred_Bitrate
        # # Initialize variables for a new episode
        # self.measured_bandwidth = Measured_Bandwidth  # Starting value (in Kbps)
        # self.previous_throughput = Previous_Throughput  # Starting value (in Kbps)
        # self.buffer_occupancy = Buffer_Occupancy  # Buffer starts empty (in seconds)
        # self.video_time = Video_Time  # Video time starts at 0 seconds
        # self.rebuffering_time = Rebuffering_Time  # No rebuffering at the start
        # self.preferred_bitrate = Preferred_Bitrate
        # #self.preferred_bitrate = self.available_bitrates[0]  # Set preferred bitrate to the first available bitrate

        # initial_state = np.array([
        #     self.measured_bandwidth,
        #     self.previous_throughput,
        #     self.buffer_occupancy,
        #     self.preferred_bitrate,
        #     self.video_time,
        #     self.rebuffering_time
        # ])

        # return initial_state
        # self.states = states
        # self.current = 0
        # return states[0]
        # return self.recv_command(0)
        return [0, 0, 0, 0, 0, 0, 0]

    def step(self, action, state):
        selected_bitrate = self.available_bitrates[action]  # Get selected bitrate (in Kbps)
        chunk_size = (selected_bitrate * 1000) * self.chunk_time / 8  # Convert to bytes

        self.measured_bandwidth = state[0]
        self.previous_throughput = state[1]
        self.buffer_size = state[2]
        self.buffer_occupancy = state[3]
        self.buffer_time = state[4]
        self.video_time = state[5]
        self.rebuffering_time = state[6]

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
        # reward = grade()/1000000 - 0.5

        # Update measured bandwidth (simulate some variability)
        # self.measured_bandwidth = np.random.randint(800, 1200)  # Simulate bandwidth variation (Kbps)

        # Determine if the video has finished
        done = self.video_time >= self.total_video_length

        # # Create the next state
        # next_state = np.array([
        #     self.measured_bandwidth,
        #     self.previous_throughput,
        #     self.buffer_occupancy,
        #     self.preferred_bitrate,
        #     self.video_time,
        #     self.rebuffering_time
        # ])

        next_state = self.recv_command(action)

        return next_state, reward, done

    def recv_command(self, action):
        messagepart = clientsocket.recv(2048).decode()

        # message += messagepart
        # if message[-1] == '\n':


        jsonargs = json.loads(messagepart)
        # message = ""
        if(jsonargs["exit"] != 0):
            return "exit"
        
        state = [jsonargs["Measured Bandwidth"], jsonargs["Previous Throughput"], jsonargs["Buffer Occupancy"]["size"], jsonargs["Buffer Occupancy"]["current"], jsonargs["Buffer Occupancy"]["time"], jsonargs["Video Time"], jsonargs["Rebuffering Time"]]
        bitrate = self.available_bitrates[action]
        # bitrate = studentcode.student_entrypoint(jsonargs["Measured Bandwidth"], jsonargs["Previous Throughput"], jsonargs["Buffer Occupancy"], jsonargs["Available Bitrates"], jsonargs["Video Time"], jsonargs["Chunk"], jsonargs["Rebuffering Time"], jsonargs["Preferred Bitrate"])
        payload = json.dumps({"bitrate" : bitrate}) + '\n'
        clientsocket.sendall(payload.encode())
        return state


# def get_states():
#     states = []
#     with open('./test_states/badtest', 'r') as json_file:
#         data = json.load(json_file)
#     intermediate_states = [data[key] for key in sorted(data.keys())]    # Outside is list, but inside are still dict
#     for state in intermediate_states:
#         states.append([state["Measured_Bandwidth"], state["Previous_Throughput"], state["Buffer_Occupancy"]["size"], state["Buffer_Occupancy"]["current"], state["Buffer_Occupancy"]["time"], state["Video_Time"], state["Rebuffering_Time"]])
#     return states

def run_simulator_code():
    output = subprocess.run(['python', 'simulator.py', './tests/badtest/trace.txt', './tests/badtest/manifest.json', ""], capture_output=True)

def train_agent(episodes, save_path="dqn_model.pth"):
    global ss
    global clientsocket

    # states = get_states()

    state_size = 7  # Change this to the number of state variables you have
    action_size = 3  # Change this to the number of available bitrates
    agent = DQNAgent(state_size, action_size)
    manifest = Manifest("./tests/badtest/manifest.json")
    simulator = Simulator(manifest)

    for e in range(episodes):
        print("Episode:", e + 1)

        #run simulator process
        simulator_thread = threading.Thread(target=run_simulator_code)
        simulator_thread.start()

        (clientsocket, address) = ss.accept()

        state = simulator.reset()
        done = False
        total_reward = 0

        while not done:
        # for state in states:
            action = agent.act(state)
            next_state, reward, done = simulator.step(action, state)  # Get the next state and reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            print("Reward: ", reward, "Time:, ", state[5])

        agent.replay(32)  # Replay experience and update the model
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Save the model every certain number of episodes
        if (e + 1) % 100 == 0:  # Save every 100 episodes
            torch.save({
                'epoch': e,
                'model_state_dict': agent.model.state_dict(),  # Save model weights
                'optimizer_state_dict': agent.optimizer.state_dict(),  # Save optimizer state
                'loss': agent.loss,  # Save the current loss
            }, save_path)
            print(f"Model saved to {save_path}")

        simulator_thread.join()

if __name__ == "__main__":
    train_agent(1000)  # Train for 1000 episodes