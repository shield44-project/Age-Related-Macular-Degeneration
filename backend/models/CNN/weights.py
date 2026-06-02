import torch

# Load the file securely on the CPU
checkpoint = torch.load("best_resnet_model.pth", map_location="cpu", weights_only=True)

# If the file contains a checkpoint dictionary, extract the state dict
state_dict = checkpoint.get("model_state_dict", checkpoint)

# Loop and print layer names with their shapes
for layer_name, weights in state_dict.items():
    print(f"Layer: {layer_name} | Shape: {weights.shape}")
