import numpy as np

def load_features(feature_path):
    """
    Load features from a file (e.g., .npy or .pth).
    """
    if feature_path.endswith('.npy'):
        features = np.load(feature_path)
    elif feature_path.endswith('.pth'):
        features = torch.load(feature_path)
    else:
        raise ValueError("Unsupported feature file format. Use .npy or .pth.")
    return features