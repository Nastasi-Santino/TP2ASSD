import numpy as np

def variable_delay(x, M0, FL, A, fs=44100, phase=0):

    M0_samples = M0 * fs / 1000
    A_samples = A * fs / 1000

    n = np.arange(len(x))
    M = M0_samples + A_samples * np.sin(2 * np.pi * FL * n / fs + phase)

    return M


def vibrato_effect(x, fs=44100):
    M = variable_delay(x, M0=5, FL=5, A=2, fs=fs)
    y = np.zeros_like(x)

    for n in range(len(x)):
        delay = M[n]

        delay_int = int(np.floor(delay))
        delay_frac = delay - delay_int

        idx0 = n - delay_int
        idx1 = idx0 - 1

        if idx1 >= 0:
            y[n] = (1 - delay_frac) * x[idx0] + delay_frac * x[idx1]
        elif idx0 >= 0:
            y[n] = x[idx0]
        else:
            y[n] = x[n]
        
    max_val = np.max(np.abs(y))
    if max_val > 1:
        y = y / max_val

    return y

def flanger_effect(x, g, fs=44100):
    M = variable_delay(x, M0=2, FL=0.5, A=2, fs=fs)
    y = np.zeros_like(x)

    for n in range(len(x)):
        delay = M[n]

        delay_int = int(np.floor(delay))
        delay_frac = delay - delay_int

        idx0 = n - delay_int
        idx1 = idx0 - 1

        if idx1 >= 0:
            delayed_sample = (1 - delay_frac) * x[idx0] + delay_frac * x[idx1]
        elif idx0 >= 0:
            delayed_sample = x[idx0]
        else:
            delayed_sample = 0

        y[n] = x[n] + g * delayed_sample

    max_val = np.max(np.abs(y))
    if max_val > 1:
        y = y / max_val
    
    return y

def chorus_effect(x, fs=44100):
        
    voices = [
        # M0_ms, FL_Hz, A_ms, phase, gain
        (22, 0.35, 4.0, 0.0,        0.35),
        (28, 0.50, 5.0, 2*np.pi/3,  0.30),
        (35, 0.25, 6.0, 4*np.pi/3,  0.25),
    ]
            
    y = x.astype(float).copy()
    
    for M0, FL, A, phase, g in voices:
        M = variable_delay(x, M0, FL, A, fs, phase)

        for n in range(len(x)):
            delay = M[n]

            delay_int = int(np.floor(delay))
            delay_frac = delay - delay_int

            idx0 = n - delay_int
            idx1 = idx0 - 1

            if idx1 >= 0:
                delayed_sample = (1 - delay_frac) * x[idx0] + delay_frac * x[idx1]
            elif idx0 >= 0:
                delayed_sample = x[idx0]
            else:
                delayed_sample = 0

            y[n] += g * delayed_sample

        
    max_val = np.max(np.abs(y))
    if max_val > 1:
        y = y / max_val

    return y

# Testbench

if __name__ == "__main__":

    import soundfile as sf

    audio, sr = sf.read('InstrumentsPSOLA/ViolinC4.wav')

    # Apply vibrato effect
    y_vibrato = vibrato_effect(audio, fs=sr)
    y_flanger = flanger_effect(audio, g=0.5, fs=sr)
    y_chorus = chorus_effect(audio, fs=sr)

    # Save the original and processed signals to WAV files

    sf.write('vibrato_effect.wav', y_vibrato, sr)
    sf.write('flanger_effect.wav', y_flanger, sr)
    sf.write('chorus_effect.wav', y_chorus, sr)