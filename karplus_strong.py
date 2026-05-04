import random
import pyaudio
import numpy as np
import time
from filtersForKS import simple_average, recursive_LP_filter, recursive_AP_filter

def signo(b, N):
    r = np.random.rand(N)
    return np.where(r < b, 1, -1)


def karplus_strong(sample_frequency, L, RL, duration, noise=0, filter=0, b=1, AP = 1, mu = 0.5):
    """
    Implementation of Karplus-Strong algorithm.
    
    Args:
        sample_frequency: Sampling frequency in Hz
        L: Delay line length (in samples)
        RL: Loss factor (0 < RL <= 1)
        duration: Duration of output in seconds
        noise: Noise distribution type (0=uniform, 1=Gaussian/normal)
        filter: Filter type (0=simple average, 1=recursive low-pass)
        b: Sign distribution parameter (0 <= b <= 1) 
        AP: Allpass filter flag (0=off, 1=on)
        mu: fractional delay parameter (0 <= mu < 1)

    Returns:
        List of output samples
    """
    # Calculate total number of samples to generate
    num_samples = int(sample_frequency * duration)
    
    # Create output array
    output = [0.0] * num_samples
    
    # Array to store v(n) - output of first adder
    v = [0.0] * num_samples
    w = [0.0] * num_samples
    
    # Initialize with white noise (excitation signal)
    if noise == 0:
        # Uniform distribution between -1 and 1
        excitation = [2.0 * (random.random() - 0.5) for _ in range(L)]
    else:
        # Gaussian/normal distribution
        excitation = [random.gauss(0, 0.5) for _ in range(L)]
    
    s = signo(b, num_samples)

    alpha = (mu - 1) / (mu + 1)

    # Generate output samples
    for n in range(num_samples):
        # Input: excitation signal for first L samples, then 0
        if n < L:
            x_n = excitation[n]
        else:
            x_n = 0.0
        
        # Get the z^-L feedback (output from L samples ago, scaled by RL)
        if n >= L:
            feedback_L = RL * output[n - L]
        else:
            feedback_L = 0.0
        
        # First adder: v(n) = x(n) + RL*y(n-L)
        v[n] = x_n + feedback_L
        
        if AP == 1:
            w[n] = recursive_AP_filter(v[n], v[n-1] if n > 0 else 0.0, w[n-1] if n > 0 else 0.0, alpha)
        else:
            w[n] = v[n]


        # Apply the internal filter
        if filter == 0:
            output[n] = s[n] * simple_average(w[n], w[n-1] if n > 0 else 0.0)
        else:
            # Apply the recursive low-pass filter
            output[n] = s[n] * recursive_LP_filter(w[n], output[n-1], 1.0, 0.15)
    
    return output


def play_audio(samples, sample_rate, duration=None):
    """
    Play audio samples using pyAudio.
    
    Args:
        samples: List of audio samples
        sample_rate: Sample rate in Hz
        duration: Optional duration to play (in seconds). If None, plays all samples.
    """
    # Convert to numpy array and normalize to int16
    audio_data = np.array(samples, dtype=np.float32)
    audio_data = np.clip(audio_data, -1.0, 1.0)  # Clip to [-1, 1]
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    # Initialize pyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    
    # Play audio
    chunk_size = 1024
    for i in range(0, len(audio_data_int16), chunk_size):
        stream.write(audio_data_int16[i:i+chunk_size].tobytes())
    
    # Close stream
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    # Musical scale frequencies (C4 major scale)
    notes = {
        "Do (C4)": 261.63,
        "Re (D4)": 293.66,
        "Mi (E4)": 329.63,
        "Fa (F4)": 349.23,
        "Sol (G4)": 392.00,
        "La (A4)": 440.00,
        "Si (B4)": 493.88,
        "Do (C5)": 523.26,
    }
    
    # Test bench parameters
    sample_rate = 44100  # Hz
    RL = 0.998  # Loss factor
    duration = 0.2  # seconds per note
    noise = 1  
    filter1 = 1
    b = 1
    AP = 1
    
    print("Karplus-Strong Algorithm - Do-Re-Mi-Fa-Sol-La-Si Test Bench")
    print("=" * 60)
    
    # Play the scale
    for idx, (note_name, frequency) in enumerate(notes.items(), 1):
        print(f"[{idx}/7] Playing {note_name} ({frequency:.2f} Hz)...")
        L = int(sample_rate / frequency - 0.5)
        mu = sample_rate / frequency - L
        samples = karplus_strong(sample_rate, L, RL, duration, noise, filter1, b, AP, mu)
        play_audio(samples, sample_rate)
        #time.sleep(0.1)  # Short pause between notes
    
    # Play the scale again
    print("\n[Repeat] Playing the scale again...")
    for idx, (note_name, frequency) in enumerate(notes.items(), 1):
        L = int(sample_rate / frequency - 0.5)
        mu = sample_rate / frequency - L
        samples = karplus_strong(sample_rate, L, RL, duration, noise, filter1, b, AP, mu)
        play_audio(samples, sample_rate)
        time.sleep(0.05)
    
    print("\nScale completed!")
