import torch
from torch_train import DQN  # Replace with your model class

# Check if CUDA is available and set the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
checkpoint = torch.load('dqn_model.pth', weights_only=True)
model = DQN(7, 3)  # Replace with your model class
model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)  # Move the model to the GPU
model.eval()  # Set to evaluation mode for inference

# def solve_diff_bitrates(state, available_bitrates):
#     # print("Available bitrates:", available_bitrates)
#     if available_bitrates == ['500000', '100000', '50000']:
#         print("Got different bitrates")
#         state[0] *= 10
#         state[1] *= 10
#         state[2] *= 10
#         state[3] *= 10
#     return state

def predict(state, available_bitrates):
    # state = solve_diff_bitrates(state, available_bitrates)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Perform inference to select bitrate
    with torch.no_grad():
        action_values = model(state_tensor)  # Get Q-values or action probabilities

    # Choose the action (bitrate) with the highest value
    selected_action = torch.argmax(action_values, dim=1).item()

    # print("Selected action:", selected_action)

    return selected_action
