import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
from data import *
from args import parse_args
import os
import numpy as np
from dp import dp  # Ensure dp is available and works on macOS

# Parse arguments
args = parse_args()

# Ensure CUDA is available or fall back to CPU (macOS typically uses CPU unless using Apple Silicon GPU with PyTorch MPS)
device = th.device("mps") if th.backends.mps.is_available() else ("cuda" if args.use_gpu and th.cuda.is_available() else "cpu")

# Load task information
primary_info = read_task_info(args.primary_path)

if args.use_related:
    related_info = read_task_info(args.related_path)
    task_steps = {**primary_info['steps'], **related_info['steps']}  # Use task_steps for step descriptions
    n_steps = {**primary_info['n_steps'], **related_info['n_steps']}
else:
    task_steps = primary_info['steps']
    n_steps = primary_info['n_steps']

# Load test dataset
test_vids = get_vids(args.val_csv_path)
testset = CrossTaskDataset(test_vids, n_steps, args.features_path, args.constraints_path)
testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False, collate_fn=lambda batch: batch)

# Use task_steps instead of n_steps
A, M = get_A(task_steps, share=args.share)

# Load saved model
net = Model(args.d, M, A, args.q)
net.load_state_dict(th.load("final_model.pth", map_location=device))
net.eval()  # Set model to evaluation mode
net.to(device)

if device != th.device("cpu"):
    A = {task: a.to(device) for task, a in A.items()}

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
                X = sample['X'].to(device)

                O = lsm(net(X, task))  # Model prediction
                y = np.zeros(O.size(), dtype=np.float32)
                dp(y, -O.cpu().numpy())  # Decode step predictions

                if task not in Y_pred:
                    Y_pred[task] = {}
                Y_pred[task][vid] = y

                # 📝 Human-readable output:
                print(f"\nPredicted Steps for Video: {vid}, Task: {task}")

                # Get step descriptions
                step_names = task_steps.get(task, ["Step " + str(i) for i in range(y.shape[1])])

                # Find frames where steps are detected
                for frame_idx, step_vector in enumerate(y):
                    detected_steps = [step_names[i] for i, val in enumerate(step_vector) if val > 0]
                    if detected_steps:
                        print(f"Frame {frame_idx}: {', '.join(detected_steps)}")

    print("\nEvaluation Complete.")

# Run evaluation
evaluate_model()
