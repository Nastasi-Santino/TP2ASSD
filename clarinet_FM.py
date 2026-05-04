import numpy as np
import pyaudio
import time

tauI = 0.5  # Time constant for intensity envelope

def A(t, T):
    Ta = T * 0.05  # Attack time
    Ts = T * 0.8  # Sustain time
    taur = T * 0.15  # Release time

    # Amplitude envelope (simple attack-decay)
    if t < Ta:
        return t / Ta  # Attack
    elif t < Ta + Ts:
        return 1  # Sustain
    else:
        return np.exp(- (t - Ta - Ts) / taur)  # Decay

def I(t, T): 
    tauI = 0.1 * T  # Time constant for intensity envelope
    return 1.5 + 2.0 * np.exp(-t / tauI)

def clarinet_FM(sample_rate, frequency, duration):
    
    num_samples = int(sample_rate * duration)

    output = [0.0] * num_samples

    carrier_freq = 1 * frequency
    modulator_freq = 2 * frequency  # Modulator frequency is an octave above
    phi_c = -np.pi / 2  # Carrier phase
    phi_m = -np.pi / 2  # Modulator phase

    for n in range(num_samples):
        t = n / sample_rate
        # Simple FM synthesis formula for a clarinet-like sound
        output[n] = A(t, duration) * np.cos(2 * np.pi * carrier_freq * t + I(t, duration) * np.cos(2 * np.pi * modulator_freq * t + phi_m) + phi_c)
    
    return output


def normalize_audio(audio_data):
    """Normalize audio to prevent clipping"""
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val * 0.95
    return audio_data


def play_audio(audio_data, sample_rate):
    """Play audio using PyAudio"""
    p = pyaudio.PyAudio()
    
    # Convert to int16 for playback
    audio_int16 = np.int16(audio_data * 32767)
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        output=True,
        frames_per_buffer=1024
    )
    
    # Play the audio
    stream.write(audio_int16.tobytes())
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    # Audio parameters
    SAMPLE_RATE = 44100
    DURATION = 1  # seconds per note (gives release envelope time to decay)
    
    # Frequencies to test (in Hz) - musical notes
    test_frequencies = [
        261.63,  # C4 (Middle C)
        293.66,  # D4
        329.63,  # E4
        349.23,  # F4
        392.00,  # G4
        440.00,  # A4 (Concert pitch)
        493.88,  # B4
        523.25,  # C5
    ]
    
    print("🎷 Clarinet FM Testbench")
    print("=" * 50)
    
    for i, freq in enumerate(test_frequencies, 1):
        print(f"[{i}/{len(test_frequencies)}] Playing {freq:.2f} Hz...", end=" ", flush=True)
        
        # Generate clarinet sound
        audio = clarinet_FM(SAMPLE_RATE, freq, DURATION)
        audio = np.array(audio)
        
        # Normalize to prevent clipping
        audio = normalize_audio(audio)
        
        # Play it
        play_audio(audio, SAMPLE_RATE)
        
        print("✓ Done")
        #time.sleep(0.1)  # Brief pause between notes
    
    print("=" * 50)
    print("🎵 Testbench complete!")