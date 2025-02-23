import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
from moviepy.editor import VideoFileClip
import librosa
from pytorch_i3d import InceptionI3d  # Use the I3D implementation from the cloned repository

# -------------------------------
# I3D Feature Extraction (Updated)
# -------------------------------
def extract_i3d_features(frames, device):
    """
    Extracts a 1024-dim RGB I3D feature vector from a list of frames using your pre-trained model.
    
    Args:
        frames (list): List of frames (as NumPy arrays) forming a clip.
        device (torch.device): Device to run the model on.
        
    Returns:
        np.ndarray: 1024-dimensional feature vector.
    """
    # 1. Instantiate the I3D model with 400 output features (to match pre-trained weights).
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    
    # 2. Load the pre-trained model weights, skipping incompatible layers.
    state_dict = torch.load("/Users/mymac/Projects/CrossTask/Models/rgb_imagenet.pt", map_location=device)
    
    # Remove the 'logits' layer from the state_dict to avoid size mismatch.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logits')}
    
    # Load the filtered state_dict.
    i3d.load_state_dict(state_dict, strict=False)
    
    # 3. Move the model to the desired device and set it to evaluation mode.
    i3d.to(device)
    i3d.eval()

    # 4. Define the transforms for preprocessing.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 5. Preprocess each frame.
    processed_frames = [transform(frame) for frame in frames]
    
    # 6. Stack frames into a tensor.
    clip = torch.stack(processed_frames, dim=0)  # Shape: [T, 3, 224, 224]
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # Rearranged to [1, 3, T, 224, 224]

    # 7. Forward pass through I3D.
    with torch.no_grad():
        features = i3d.extract_features(clip)
    
    # 8. Flatten the features to a 1D array.
    return features.squeeze().cpu().numpy().reshape(-1)  # Flatten to 1D array

# -------------------------------
# ResNet-152 Feature Extraction
# -------------------------------
def extract_resnet_features(frame, device):
    """
    Extracts a 2048-dim feature vector from a single frame using ResNet-152.
    
    Args:
        frame (np.ndarray): A single video frame in BGR format.
        device (torch.device): Device to run the model on.
        
    Returns:
        np.ndarray: 2048-dimensional feature vector.
    """
    resnet = models.resnet152(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(input_tensor).squeeze()
    return features.cpu().numpy().reshape(-1)  # Ensure 1D array

# -------------------------------
# Audio Feature Extraction (Updated with Fallback)
# -------------------------------
def extract_audio_features(audio_signal, sr, device):
    """
    Extracts a 128-dim audio feature vector from a 1-second audio segment.
    Uses VGGish if available; otherwise, falls back to MFCC features using librosa.
    
    Args:
        audio_signal (np.ndarray): 1D audio signal for 1 second.
        sr (int): Sample rate of the audio (should be 16000 Hz for VGGish).
        device (torch.device): Device to run the model on.
        
    Returns:
        np.ndarray: 128-dimensional audio feature vector.
    """
    try:
        # Try using VGGish if available.
        import torchvggish
        import torchvggish.vggish as vggish
        import torchvggish.vggish_input as vggish_input

        # Initialize VGGish.
        model = vggish.VGGish()
        model.preprocess = False  # We'll do preprocessing ourselves.
        model.to(device)
        model.eval()

        # Convert the waveform into log mel spectrogram examples.
        mel_examples = vggish_input.waveform_to_examples(audio_signal, sr)
        # mel_examples shape: [num_examples, 96, 64]
        
        # Convert to tensor and add channel dimension.
        input_tensor = torch.tensor(mel_examples, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            embeddings = model(input_tensor)  # Output shape: [batch_size, 128]
        
        # Average embeddings across all examples.
        return torch.mean(embeddings, dim=0).cpu().numpy()
    except ImportError:
        # Fallback to librosa MFCC features if VGGish is not available.
        print("Warning: torchvggish not found. Falling back to librosa MFCC features.")
        mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=20)  # Extract 20 MFCC features.
        return np.mean(mfcc, axis=1)  # Average over time to get a 20-dimensional feature vector.

# -------------------------------
# Main Feature Extraction Function
# -------------------------------
def extract_video_features(video_path, output_path, use_gpu=False):
    """
    Extracts a 3200-dim feature per second from the input video by concatenating:
      - 1024-dim I3D features (columns 0-1023),
      - 2048-dim ResNet-152 features (columns 1024-3071),
      - 128-dim Audio VGG features (columns 3072-3199).
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the features (e.g., features.npy).
        use_gpu (bool): Whether to use GPU.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Open video.
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps  # in seconds

    # Load audio via moviepy.
    clip = VideoFileClip(video_path)
    audio = clip.audio

    # Check if audio is loaded and has a valid duration.
    if audio is None or not hasattr(audio, 'duration') or audio.duration is None:
        print("Warning: Audio is not available or has no duration. Using silent audio.")
        audio_array = np.zeros((int(duration) * 16000,))  # Generate silent audio
    else:
        # Extract audio as a NumPy array.
        sr = 16000  # Desired sample rate.
        try:
            audio_array = audio.to_soundarray(fps=sr)  # Extract audio as a NumPy array
            if audio_array.ndim == 2:  # If stereo, convert to mono
                audio_array = np.mean(audio_array, axis=1)
        except Exception as e:
            print("Error extracting audio:", e)
            print("Using silent audio as fallback.")
            audio_array = np.zeros((int(duration) * 16000,))  # Generate silent audio

    # Resample audio if necessary.
    if len(audio_array) < sr * duration:
        pad_width = int(sr * duration) - len(audio_array)
        audio_array = np.pad(audio_array, (0, pad_width), mode='constant')

    features_per_second = []

    # Process video second by second.
    for sec in range(int(duration)):
        start_frame = int(sec * video_fps)
        end_frame = int((sec + 1) * video_fps)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for idx in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            continue

        # --- I3D Features ---
        indices = np.linspace(0, len(frames) - 1, num=16, dtype=int)
        i3d_clip = [frames[i] for i in indices]
        i3d_feature = extract_i3d_features(i3d_clip, device)

        # --- ResNet-152 Features ---
        central_frame = frames[len(frames) // 2]
        resnet_feature = extract_resnet_features(central_frame, device)

        # --- Audio Features ---
        # Extract audio corresponding to this second.
        audio_start = int(sec * sr)
        audio_end = int((sec + 1) * sr)
        audio_segment = audio_array[audio_start:audio_end]
        if len(audio_segment) < sr:
            pad_width = sr - len(audio_segment)
            audio_segment = np.pad(audio_segment, (0, pad_width), mode='constant')
        audio_feature = extract_audio_features(audio_segment, sr, device)

        # Concatenate to form 3200-dim vector.
        feature_vector = np.concatenate([i3d_feature, resnet_feature, audio_feature])
        features_per_second.append(feature_vector)
        print(f"Processed second {sec+1} / {int(duration)}")
    
    cap.release()
    features_array = np.array(features_per_second)
    np.save(output_path, features_array)
    print(f"\nSaved extracted features to {output_path}")

# -------------------------------
# Script Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 3200-dim features per second from a video (I3D, ResNet-152, Audio VGG)."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("output_path", type=str, help="Path to save the features (e.g., features.npy)")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for feature extraction if available")
    args = parser.parse_args()
    
    extract_video_features(args.video_path, args.output_path, args.use_gpu)