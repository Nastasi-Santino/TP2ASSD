import numpy as np

FluteAk = [1, 1.0895, 1.7724, 1.6287, 0.5506, 0.4013, 0.1149, 0.6623, 0.4113, 0.1623, 0.0713]
Aflute = 0.2
kflute = 1
Dflute = 0
Rflute = 0.2
Sflute = 1 - Aflute - Dflute - Rflute
A0flute = 0.6236
alpha_flute = 0.2029

CelloAk = [1, 0.2080, 0.2705, 0.3558, 0.2211, 0.1083, 0.0742, 0.0905, 0.0367, 0, 0, 0.0327]
Acello = 0.086
kcello = 1.62
Dcello = 0.0633
Rcello = 0.23
Scello = 1 - Acello - Dcello - Rcello
A0cello = 0.3713
alpha_cello = 0.1169

PianoAk = [1.0000, 9.8653, 1.2526, 0.5044, 1.2690, 0.8709, 4.9047, 0.4080, 1.2052, 2.3297, 1.0400, 0.7977, 0.4121, 1.4634, 0.5920]
Apiano = 0.0251
kpiano = 1
Dpiano = 0
Spiano = 0
Rpiano = 1 - Apiano - Dpiano - Spiano
A0piano = 1
alpha_piano = 0

TrumpetAk = [1, 3.9291, 6.8011, 9.9342, 9.7906, 6.9109, 4.1108, 2.7408, 2.5192, 1.8966,1.5985, 1.4077, 1.0411, 0.8207]
Atrumpet = 0.0414
ktrumpet = 1/0.92
Dtrumpet = 0.1269
Rtrumpet = 0.0636
Strumpet = 1 - Atrumpet - Dtrumpet - Rtrumpet
A0trumpet = 1.0435
alpha_trumpet = -0.0616

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
    elif instrument == 'cello':
        Ak = CelloAk  # Amplitude coefficients for the cello
    elif instrument == 'piano':
        Ak = PianoAk  # Amplitude coefficients for the piano
    elif instrument == 'trumpet':
        Ak = TrumpetAk  # Amplitude coefficients for the trumpet

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
    elif instrument == 'cello':
        Ak = CelloAk  # Amplitude coefficients for the cello
        A = Acello
        D = Dcello
        S = Scello
        R = Rcello
        A0 = A0cello
        alpha = alpha_cello
        k = kcello
    elif instrument == 'piano':
        Ak = PianoAk  # Amplitude coefficients for the piano
        A = Apiano
        D = Dpiano
        S = Spiano
        R = Rpiano
        A0 = A0piano
        alpha = alpha_piano
        k = kpiano
    elif instrument == 'trumpet':
        Ak = TrumpetAk  # Amplitude coefficients for the trumpet
        A = Atrumpet
        D = Dtrumpet
        S = Strumpet
        R = Rtrumpet
        A0 = A0trumpet
        alpha = alpha_trumpet
        k = ktrumpet
    
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
    elif instrument == 'cello':
        Ak = CelloAk  # Amplitude coefficients for the cello
        # Load custom envelope from CSV
        envelope_data = np.loadtxt('envelopes/CelloC3_envelope.csv', delimiter=',')
    elif instrument == 'piano':
        Ak = PianoAk  # Amplitude coefficients for the piano
        # Load custom envelope from CSV
        envelope_data = np.loadtxt('envelopes/PianoC3_envelope.csv', delimiter=',')
    elif instrument == 'trumpet':
        Ak = TrumpetAk  # Amplitude coefficients for the trumpet
        # Load custom envelope from CSV
        envelope_data = np.loadtxt('envelopes/TrumpetC4_envelope.csv', delimiter=',')
        
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
    import soundfile as sf
    
    # ========== G3 Major Scale Generation ==========
    # G3 major scale frequencies
    g_major_scale = [
        ('G3', 196.00),
        ('A3', 220.00),
        ('B3', 246.94),
        ('C4', 261.63),
        ('D4', 293.66),
        ('E4', 329.63),
        ('F#4', 369.99),
        ('G4', 392.00),
    ]
    
    note_duration = 0.12  # 120 ms per note
    amplitude = 0.5
    fs = 44100  # Sample rate
    
    # Generate all notes
    scale_audio = np.array([])
    
    # print("Generating G3 Major Scale:")
    # for note_name, frequency in g_major_scale:
    #     # Generate note using ASDR envelope
    #     note = additive_synthesis_with_envelope(frequency, note_duration, amplitude, fs=fs, instrument='flute')
    #     scale_audio = np.concatenate([scale_audio, note])
    #     print(f"  ✓ {note_name} ({frequency:.2f} Hz)")
    
    # # Save scale to WAV file
    # sf.write('G3_major_scale.wav', scale_audio, fs)
    # print(f"\n✓ G3 Major Scale saved as 'G3_major_scale.wav'")
    # print(f"  Total duration: {len(scale_audio) / fs:.2f} seconds")
    
    # ========== Individual Examples ==========
    print("\nGenerating individual examples:")
    f0 = 261.63  # Frequency of C4
    duration = 2.0  # Duration in seconds
    Amplitud = 0.5  # Amplitude
    synthesized_note = additive_synthesis_without_envelope(f0, duration, Amplitud, fs=44100, instrument='trumpet')
    synthesized_note_with_ASDR = additive_synthesis_with_ASDR(f0, duration, Amplitud, fs=44100, instrument='trumpet')
    synthesized_note_with_envelope = additive_synthesis_with_envelope(f0, duration, Amplitud, fs=44100, instrument='trumpet')

    # Save the synthesized notes to WAV files
    sf.write('synthesized_C4_trumpet.wav', synthesized_note, 44100)
    sf.write('synthesized_C4_trumpet_with_ASDR.wav', synthesized_note_with_ASDR, 44100)
    sf.write('synthesized_C4_trumpet_with_envelope.wav', synthesized_note_with_envelope, 44100)
    print("  ✓ Individual notes saved")