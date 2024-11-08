import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import random
import rl_simulator
import time
from torch_utils import *

# # Quick-and-dirty TCP Server:
# # ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
# ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# ss.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# ss.bind(('localhost', 6000))
# ss.listen(10)
# # print('Waiting for simulator')

# class TransformerDQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(TransformerDQN, self).__init__()
#         config = BertConfig(
#             hidden_size=256 ,
#             num_hidden_layers=4,
#             num_attention_heads=8,
#             intermediate_size=512,
#             max_position_embeddings=state_size,
#         )
#         self.transformer = BertModel(config)
#         self.fc = nn.Linear(256, action_size)

#     def forward(self, x):
#         # Ensure x has the shape [batch_size, seq_length, hidden_size]
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)  # Add batch dimension if missing
#         x = self.transformer(inputs_embeds=x).last_hidden_state
#         x = x[:, 0, :]  # Use the representation of the [CLS] token
#         return self.fc(x)

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
                    if done:
                        agent.update_target_model()

                agent.replay(64)  # Replay experience and update the model
                print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Time: {time.time() - start_time:.2f}")

                # Save the model every certain number of episodes
                # if (e + 1) % 100 == 0:  # Save every 100 episodes
                # print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
            print(f"Task: {task}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            print("In Epoch: ", ep + 1)
            torch.save({
                'model_state_dict': agent.model.state_dict(),  # Save model weights
                'optimizer_state_dict': agent.optimizer.state_dict(),  # Save optimizer state
                'loss': agent.loss,  # Save the current loss
            }, save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    total_time_start = time.time()
    train_agent(100, 100)  # Train 100 round, train 100 episodes in different tests each round
    print(f"Total training time: {time.time() - total_time_start:.2f} seconds")