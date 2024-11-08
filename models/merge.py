import torch
import torch.nn as nn

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

checkpoint1 = torch.load('./1/dqn_model.pth', weights_only=True)
checkpoint2 = torch.load('./2/dqn_model_from2.pth', weights_only=True)

# Assume model1 and model2 are your trained models
model1 = DQN(10, 3)  # Your first model
model2 = DQN(10, 3)  # Your second model

model1.load_state_dict(checkpoint1['model_state_dict'])
model2.load_state_dict(checkpoint2['model_state_dict'])

# Save the models in a dictionary
models = {
    'model1': model1.state_dict(),
    'model2': model2.state_dict()
}

# Save the dictionary to a .pth file
torch.save(models, 'models.pth')