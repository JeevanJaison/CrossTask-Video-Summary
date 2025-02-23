import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
from data import *
from args import parse_args
import os
import numpy as np
from dp import dp  # Ensure dp is available

# Parse arguments
args = parse_args()

# Disable GPU usage (Fix for Mac)
args.use_gpu = False  # Force CPU mode

# Load task information
primary_info = read_task_info(args.primary_path)

if args.use_related:
    related_info = read_task_info(args.related_path)
    task_steps = {**primary_info['steps'], **related_info['steps']}  
    n_steps = {**primary_info['n_steps'], **related_info['n_steps']}
else:
    task_steps = primary_info['steps']
    n_steps = primary_info['n_steps']

# Load test dataset
test_vids = get_vids(args.val_csv_path)

# Fix constraint file path issue
constraint_files = []
missing_files = []
for vid in test_vids:
    constraint_file = os.path.join(args.constraints_path, vid)
    if not constraint_file.endswith("_f1.csv"):
        constraint_file += "_f1.csv"
    
    constraint_files.append(constraint_file)

    # Check if the file exists
    if not os.path.exists(constraint_file):
        missing_files.append(constraint_file)

# Print missing constraint files
if missing_files:
    print("\n⚠️ The following constraint files are missing:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n⚠️ ERROR: Missing constraint files. Please check your dataset.")
    exit(1)  # Stop execution since necessary files are missing

# Load dataset
testset = CrossTaskDataset(test_vids, n_steps, args.features_path, args.constraints_path)
testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False, collate_fn=lambda batch: batch)

# Compute step-to-component matrix
A, M = get_A(task_steps, share=args.share)

# Initialize model
net = Model(args.d, M, A, args.q)

# Load pre-trained model weights, skipping mismatched layers
model_dict = net.state_dict()
pretrained_dict = th.load("final_model.pth", map_location=th.device("cpu"))

# Only load matching layers
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

# Update the model state dict
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict, strict=False)

# Reinitialize FC layer if necessary
expected_fc_size = 128  # Adjust this if needed
if net.fc.weight.shape[0] != expected_fc_size:
    print(f"⚠️ Reinitializing FC layer: {net.fc.weight.shape[0]} → {expected_fc_size}")
    net.fc = nn.Linear(3200, expected_fc_size)  # Ensure correct shape
    net.fc.bias = nn.Parameter(th.zeros(expected_fc_size))

net.eval()  # Set model to evaluation mode

# Softmax for prediction
lsm = nn.LogSoftmax(dim=1)

# Evaluation function
def evaluate_model():
    Y_pred = {}

    with th.no_grad():  # Disable gradient calculation for evaluation
        for batch in testloader:
            for sample in batch:
                vid = sample['vid']
                task = sample['task']
                X = sample['X']

                O = lsm(net(X, task))  # Model prediction
                y = np.zeros(O.size(), dtype=np.float32)
                dp(y, -O.cpu().numpy())  # Decode step predictions

                if task not in Y_pred:
                    Y_pred[task] = {}
                Y_pred[task][vid] = y

                # Debugging output
                print(f"\nPredicted Steps for Video: {vid}, Task: {task}")

    print("\nEvaluation Complete.")

# Run evaluation
evaluate_model()
