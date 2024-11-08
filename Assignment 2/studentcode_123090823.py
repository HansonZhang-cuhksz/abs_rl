import torch
import torch.nn as nn

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

# Load the trained model
checkpoint = torch.load('dqn_model.pth', weights_only=True)
model1 = DQN(10, 3)
model2 = DQN(10, 3)
model1.load_state_dict(checkpoint['model1'])
model2.load_state_dict(checkpoint['model2'])
model1.eval()
model2.eval()

bitrate = 0 #used to save previous bitrate

def student_entrypoint(Measured_Bandwidth, Previous_Throughput, Buffer_Occupancy, Available_Bitrates, Video_Time, Chunk, Rebuffering_Time, Preferred_Bitrate):
    #student can do whatever they want from here going forward
    global bitrate
    R_i = list(Available_Bitrates.items())
    R_i.sort(key=lambda tup: tup[1] , reverse=True)
    bitrate = rl_based(Measured_Bandwidth, Previous_Throughput, Buffer_Occupancy, Video_Time, Rebuffering_Time, R_i)
    return bitrate

def rl_based(m_band, prev_throughput, buf_occ, current_time, rebuff_time, R_i):
    state = [m_band, prev_throughput, buf_occ["size"], buf_occ["current"], buf_occ["time"], current_time, rebuff_time] + [int(btr) for btr, _ in R_i]
    action = predict(state, [btr for btr, _ in R_i])
    return R_i[action][0]

def predict(state, available_bitrates):
    model = model1 if available_bitrates == ['5000000', '1000000', '500000'] else model2
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # Perform inference to select bitrate
    with torch.no_grad():
        action_values = model(state_tensor)  # Get Q-values or action probabilities
    # Choose the action (bitrate) with the highest value
    selected_action = torch.argmax(action_values, dim=1).item()
    return selected_action