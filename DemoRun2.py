import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
import os
import numpy as np
from dp import dp  # Ensure dp is available and works on macOS
from data import get_A  # Replace 'utils' with the correct file/module name
from args import parse_args  # Import parse_args from args.py


# Device setup: macOS uses MPS if available, otherwise CUDA/CPU
device = th.device("mps") if th.backends.mps.is_available() else ("cuda" if th.cuda.is_available() else "cpu")

# Function to load custom features
def load_custom_features(feature_path):
    """
    Load your precomputed features from the specified path.
    Assumes each feature file is in .npy format for simplicity.
    """
    features = np.load(feature_path)  # Replace with the correct loading logic if not .npy
    return th.tensor(features, dtype=th.float32).to(device)

# Load task information (adapt if needed)
task_steps = {"task_name": ["Step 1", "Step 2", "Step 3"]}  # Replace with your actual task steps

# Define the adjacency matrix and step mapping
A, M = get_A(task_steps, share=False)  # Adjust `share` based on your needs

# Load saved model
args = parse_args()  # Ensure args contain relevant paths and settings
net = Model(args.d, M, A, args.q)
net.load_state_dict(th.load("final_model.pth", map_location=device))
net.eval()  # Set model to evaluation mode
net.to(device)

# Ensure A is on the correct device
if device != th.device("cpu"):
    A = {task: a.to(device) for task, a in A.items()}

# Softmax for prediction
lsm = nn.LogSoftmax(dim=1)

# Evaluation function for custom features
def evaluate_custom_features(feature_path, task_name="task_name"):
    """
    Evaluate the model using only the custom features provided.
    """
    custom_features = load_custom_features(feature_path)
    task_steps = {"Step 1", "Step 2", "Step 3"}  # Adjust this to your task's steps.

    with th.no_grad():
        # Model prediction
        O = lsm(net(custom_features, task_name))  # `task_name` is used to get task-specific matrix
        y = np.zeros(O.size(), dtype=np.float32)
        dp(y, -O.cpu().numpy())  # Decode step predictions

        # Human-readable output
        print(f"\nPredicted Steps for Task: {task_name}")

        # Get step descriptions
        step_names = task_steps.get(task_name, ["Step " + str(i) for i in range(y.shape[1])])

        # Find frames where steps are detected
        for frame_idx, step_vector in enumerate(y):
            detected_steps = [step_names[i] for i, val in enumerate(step_vector) if val > 0]
            if detected_steps:
                print(f"Frame {frame_idx}: {', '.join(detected_steps)}")

    print("\nEvaluation Complete.")

# Run evaluation for custom features
feature_path = "/features.npy"  # Replace with the actual path to your features
evaluate_custom_features(feature_path)
