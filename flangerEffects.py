import numpy as np

def variable_delay(x,M0, FL, A, fs=44100):
    """
    Variable delay line for flanger effect.

    Implements: M(n) = M0 + A * sin(2 * pi * FL * n / fs)
    Parameters:
    x: input audio signal
    M0: base delay in samples
    FL: modulation frequency in Hz
    A: modulation depth in samples
    fs: sampling rate (default 44100 Hz)
    Returns:
    M: array of delay values for each sample
    """
    n = np.arange(0, len(x))  # Indices for the input signal
    M = M0 + A * np.sin(2 * np.pi * FL * n / fs)
    return M.astype(int)