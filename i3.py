import torch
import numpy as np
import torchvision.transforms as transforms
from pytorch_i3d import InceptionI3d
  # Ensure your I3D implementation is available

def extract_i3d_features(frames, device):
    """
    Extracts a 1024-dim RGB I3D feature vector from a list of frames.
    
    Args:
        frames (list): List of frames (as NumPy arrays) forming a clip.
        device (torch.device): Device to run the model on.
        
    Returns:
        np.ndarray: 1024-dimensional feature vector.
    """
    # 1. Instantiate the I3D model with 1024 output features.
    #    This requires that the final logits layer is replaced to output 1024-dim features.
    i3d = InceptionI3d(num_classes=1024, in_channels=3)
    
    # 2. Load the pre-trained model weights from your file.
    #    Ensure the checkpoint file "rgb_imagnet.pt" is in the correct path.
    state_dict = torch.load("/Users/mymac/Projects/CrossTask/Models/rgb_imagenet.pt", map_location=device)
    i3d.load_state_dict(state_dict)
    
    # 3. Move the model to the desired device and set to evaluation mode.
    i3d.to(device)
    i3d.eval()

    # 4. Define the transforms to preprocess the frames as required by I3D.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to the expected input size.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 5. Preprocess each frame.
    processed_frames = []
    for frame in frames:
        processed_frame = transform(frame)
        processed_frames.append(processed_frame)
    
    # 6. Stack frames into a tensor.
    #    I3D expects input of shape [batch_size, channels, time, height, width].
    clip = torch.stack(processed_frames, dim=0)  # Shape: [T, 3, 224, 224]
    # Rearrange dimensions to [1, 3, T, 224, 224]
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # 7. Forward pass through the I3D model to extract features.
    with torch.no_grad():
        features = i3d.extract_features(clip)
    
    # 8. Squeeze out the batch dimension and convert to NumPy.
    feature_vector = features.squeeze(0).cpu().numpy()
    return feature_vector
print("Over")
