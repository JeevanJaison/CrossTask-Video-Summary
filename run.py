import torch
import numpy as np
from model import Model  # Assuming the model is defined in model.py
from utils import load_features  # Assuming you have a utility function to load features

# Load the pre-trained model
def load_model(model_path, device):
    model = Model()  # Initialize your model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the pre-computed features
def load_video_features(feature_path):
    features = load_features(feature_path)  # Load the features from the file
    return features

# Predict the video summary
def predict_summary(model, features, device):
    with torch.no_grad():
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(features)
        summary = output.squeeze(0).cpu().numpy()
    return summary

# Main function to generate the video summary
def generate_video_summary(model_path, feature_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load the model
    model = load_model(model_path, device)
    
    # Load the video features
    features = load_video_features(feature_path)
    
    # Predict the summary
    summary = predict_summary(model, features, device)
    
    # Post-process the summary (e.g., thresholding to get binary labels)
    summary = (summary > 0.5).astype(int)  # Example thresholding
    
    return summary

# Example usage
if __name__ == "__main__":
    model_path = "/Users/mymac/Projects/CrossTask/final_model1.pth"
    feature_path = "/Users/mymac/Projects/CrossTask/f1.npy"
    
    summary = generate_video_summary(model_path, feature_path)
    print("Predicted Video Summary:", summary)