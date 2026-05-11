import numpy as np

FluteAk = [1, 1.0895, 1.7724, 1.6287, 0.5506, 0.4013, 0.1149, 0.6623, 0.4113, 0.1623, 0.0713]
Aflute = 0.2
kflute = 1
Dflute = 0
Rflute = 0.2
Sflute = 1 - Aflute - Dflute - Rflute
A0flute = 0.6236
alpha_flute = 0.2029

def additive_synthesis_without_envelope(f0, duration, Amplitud, fs=44100, instrument='flute'):
    """
    Generate a note using additive synthesis.
    
    Parameters:
    f0: fundamental frequency (Hz)
    duration: duration of the note (seconds)
    fs: sampling rate (default 44100 Hz)
    instrument: type of instrument ('flute' supported)
    
    Returns:
    y: synthesized audio signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
    y = np.zeros_like(t)  # Initialize output signal

    if instrument == 'flute':
        Ak = FluteAk  # Amplitude coefficients for the flute

    for k in range(1, len(Ak) + 1):
        y += Ak[k - 1] * np.sin(2 * np.pi * k * f0 * t)

    y = y / np.max(np.abs(y))
    y = Amplitud * y

    return y

def additive_synthesis_with_ASDR(f0, duration, Amplitud, fs=44100, instrument='flute'):
    """
    Generate a note using additive synthesis with ASDR envelope.
    
    Parameters:
    f0: fundamental frequency (Hz)
    duration: duration of the note (seconds)
    Amplitud: peak amplitude of the note
    fs: sampling rate (default 44100 Hz)
    instrument: type of instrument ('flute' supported)
    
    Returns:
    y: synthesized audio signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
    y = np.zeros_like(t)  # Initialize output signal

    if instrument == 'flute':
        Ak = FluteAk  # Amplitude coefficients for the flute
        A = Aflute
        D = Dflute
        S = Sflute
        R = Rflute
        A0 = A0flute
        alpha = alpha_flute
        k = kflute

    for i in range(1, len(Ak) + 1):
        y += Ak[i - 1] * np.sin(2 * np.pi * i * f0 * t)

    y = y / np.max(np.abs(y))  # Normalize to [-1, 1]

    # Generate ASDR envelope
    env = np.zeros_like(t)
    attack_time = A * duration
    decay_time = D * duration
    sustain_time = S * duration
    release_time = R * duration

    attack_samples = int(attack_time * fs)
    decay_samples = int(decay_time * fs)
    sustain_samples = int(sustain_time * fs)
    release_start = attack_samples + decay_samples + sustain_samples
    release_samples = len(env) - release_start  # Calculate remaining samples

    env[:attack_samples] = np.linspace(0, k * A0, attack_samples)  # Attack phase
    env[attack_samples:attack_samples + decay_samples] = np.linspace(k * A0, A0, decay_samples)  # Decay phase
    env[attack_samples + decay_samples:release_start] = np.linspace(A0, A0 + alpha * sustain_time, sustain_samples) # Sustain phase
    env[release_start:] = np.linspace(A0 + alpha * sustain_time, 0, release_samples)  # Release phase

    y *= env  # Apply envelope to the signal
    y *= Amplitud / np.max(np.abs(y))
    
    return y

def additive_synthesis_with_envelope(f0, duration, Amplitud, fs=44100, instrument='flute'):
    """
    Generate a note using additive synthesis with a custom envelope.
    
    Parameters:
    f0: fundamental frequency (Hz)
    duration: duration of the note (seconds)
    Amplitud: peak amplitude of the note
    fs: sampling rate (default 44100 Hz)
    instrument: type of instrument ('flute' supported)
    
    Returns:
    y: synthesized audio signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
    y = np.zeros_like(t)  # Initialize output signal

    if instrument == 'flute':
        Ak = FluteAk  # Amplitude coefficients for the flute
        # Load custom envelope from CSV
        envelope_data = np.loadtxt('envelopes/FluteC4_envelope.csv', delimiter=',')
        
        # Resample envelope to match the current duration
        envelope_original_time = np.linspace(0, 1, len(envelope_data))
        envelope = np.interp(np.linspace(0, 1, len(t)), envelope_original_time, envelope_data)

    for i in range(1, len(Ak) + 1):
        y += Ak[i - 1] * np.sin(2 * np.pi * i * f0 * t)

    y = y / np.max(np.abs(y))  # Normalize to [-1, 1]
    y *= envelope  # Apply custom envelope
    y *= Amplitud / np.max(np.abs(y))
    
    return y

# Test Bench

if __name__ == "__main__":
    # Example usage: Generate a C4 note (261.63 Hz) for 2 seconds with amplitude 0.5
    f0 = 261.63  # Frequency of C4
    duration = 2.0  # Duration in seconds
    Amplitud = 0.5  # Amplitude
    synthesized_note = additive_synthesis_without_envelope(f0, duration, Amplitud, fs=44100, instrument='flute')
    synthesized_note_with_ASDR = additive_synthesis_with_ASDR(f0, duration, Amplitud, fs=44100, instrument='flute')
    synthesized_note_with_envelope = additive_synthesis_with_envelope(f0, duration, Amplitud, fs=44100, instrument='flute')

    # Save the synthesized note to a WAV file
    import soundfile as sf
    sf.write('synthesized_C4_flute.wav', synthesized_note, 44100)
    sf.write('synthesized_C4_flute_with_ASDR.wav', synthesized_note_with_ASDR, 44100)
    sf.write('synthesized_C4_flute_with_envelope.wav', synthesized_note_with_envelope, 44100)