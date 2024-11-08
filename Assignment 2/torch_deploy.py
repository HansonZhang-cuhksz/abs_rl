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

def predict(state, available_bitrates):
    model = model1 if available_bitrates == ['5000000', '1000000', '500000'] else model2
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # Perform inference to select bitrate
    with torch.no_grad():
        action_values = model(state_tensor)  # Get Q-values or action probabilities
    # Choose the action (bitrate) with the highest value
    selected_action = torch.argmax(action_values, dim=1).item()
    return selected_action