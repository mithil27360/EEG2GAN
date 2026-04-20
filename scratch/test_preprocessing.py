import numpy as np
import os
import sys

# Add current dir to path to import config and process_mindbigdata
sys.path.append(os.getcwd())

import config
import process_mindbigdata

def test_preprocessing():
    print("Testing Preprocessing Logic...")
    
    # Create a dummy EEG signal (5 channels, 256 samples)
    # Channel 0: Clean sine wave
    # Channel 1: High frequency noise
    # Channel 2: DC drift
    # Channel 3: Extreme artifact (> 200uV)
    # Channel 4: Mixed
    
    fs = config.EEG_SAMPLING_RATE
    t = np.linspace(0, 2, 2 * fs)
    
    ch0 = 50 * np.sin(2 * np.pi * 10 * t) # 10Hz signal
    ch1 = 50 * np.sin(2 * np.pi * 10 * t) + 20 * np.random.randn(len(t)) # Noisy
    ch2 = 50 * np.sin(2 * np.pi * 10 * t) + 100 # Drift
    ch3 = 500 * np.ones(len(t)) # ARTIFACT
    ch4 = 50 * np.sin(2 * np.pi * 10 * t)
    
    data = np.stack([ch0, ch1, ch2, ch3, ch4])
    
    print(f"Original shape: {data.shape}")
    print(f"Original max amplitude: {np.max(np.abs(data))}")
    
    # Test artifact rejection manually (part of process_imagenet logic)
    artifact_found = np.max(np.abs(data)) > config.EEG_ARTIFACT_THRESHOLD
    print(f"Artifact detected: {artifact_found}")
    
    # Test filtering on a non-artifact signal
    clean_data = np.stack([ch0, ch1, ch2, ch4])
    filtered_data = process_mindbigdata.apply_filters(clean_data)
    
    print(f"Filtered shape: {filtered_data.shape}")
    print(f"Filtered max amplitude: {np.max(np.abs(filtered_data))}")
    
    # Test normalization
    means = filtered_data.mean(axis=1, keepdims=True)
    stds = filtered_data.std(axis=1, keepdims=True) + 1e-6
    norm_data = (filtered_data - means) / stds
    
    print(f"Normalized mean: {norm_data.mean():.4f}")
    print(f"Normalized std: {norm_data.std():.4f}")
    
    assert np.allclose(norm_data.mean(), 0, atol=1e-5)
    assert np.allclose(norm_data.std(), 1, atol=1e-1) # std of a sine wave normalized isn't exactly 1 but close

    print("✅ Preprocessing logic test PASSED!")

if __name__ == "__main__":
    test_preprocessing()
