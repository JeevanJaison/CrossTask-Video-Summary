def extract_audio_features(audio_signal, sr, device):
    """
    Extracts a 128-dim audio feature vector from a 1-second audio segment using VGGish.
    
    Args:
        audio_signal (np.ndarray): 1D audio signal for 1 second.
        sr (int): Sample rate of the audio (should be 16000 Hz for VGGish).
        device (torch.device): Device to run the model on.
        
    Returns:
        np.ndarray: 128-dimensional audio feature vector.
    """
    import torchvggish
    import torchvggish.vggish as vggish
    import torchvggish.vggish_input as vggish_input

    # Initialize the VGGish model.
    model = vggish.VGGish()
    model.preprocess = False  # We'll do preprocessing ourselves.
    model.to(device)
    model.eval()

    # Convert the raw waveform into log mel spectrogram examples.
    # VGGish expects 16 kHz audio, so ensure sr is 16000.
    mel_examples = vggish_input.waveform_to_examples(audio_signal, sr)
    # mel_examples shape is [num_examples, 96, 64]
    
    # Convert the mel spectrogram to a tensor and add a channel dimension.
    # Final shape: [batch_size, channels, 96, 64]
    input_tensor = torch.tensor(mel_examples, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Forward pass through VGGish to obtain embeddings.
    with torch.no_grad():
        embeddings = model(input_tensor)  # Output shape: [batch_size, 128]
    
    # Average embeddings across all examples to produce a single 128-dim feature vector.
    feature_vector = torch.mean(embeddings, dim=0).cpu().numpy()
    return feature_vector
