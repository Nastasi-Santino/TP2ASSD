import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np


def spectogram(data, fs, nSegment=256, overlapPorcentage=None, window='hann'):
    """
    Compute and plot the spectrogram of a signal.

    Parameters:
    data (array-like): Input signal.
    fs (float): Sampling frequency of the signal.
    nSegment (int): Length of each segment for the spectrogram. Default is 256.
    overlapPorcentage (float): Percentage of overlap between segments. Default is None (no overlap).
    window (str or tuple): Desired window to use. Default is 'hann'.

    Returns:
    None
    """
    
    # Set default overlap to 0 if not specified
    if overlapPorcentage is None:
        noverlap = 0
    else:
        noverlap = int(nSegment * overlapPorcentage / 100)

    # Stride between segment starts: distance to skip when moving to next segment
    stride = nSegment - noverlap
    
    # Generate all segment starting indices
    segment_starts = range(0, len(data) - nSegment + 1, stride)
    
    # Extract overlapping segments
    segments = np.array([data[i:i + nSegment] for i in segment_starts])

    # Compute the fft for each segment
    try:
        window_func = signal.get_window(window, nSegment)
    except ValueError:
        raise ValueError("Invalid window specification")

    ffts = np.fft.fft(segments * window_func, axis=1)
    
    # Compute magnitude spectrum and keep only positive frequencies
    magnitude = np.abs(ffts)
    magnitude = magnitude[:, :nSegment // 2]  # Keep only positive frequencies
    
    # Frequency axis (Hz)
    freqs = np.fft.fftfreq(nSegment, 1/fs)[:nSegment // 2]
    
    # Time axis (seconds) - center of each segment
    times = np.array([i / fs for i in segment_starts])
    
    # Create the spectrogram plot
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, magnitude.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


# Testbench

if __name__ == "__main__":
    # Generate an FM (Frequency Modulated) signal with time-varying frequency
    fs = 1000  # Sampling frequency (Hz)
    duration = 3  # Duration (seconds)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
    
    # FM Signal: Carrier frequency modulated by a sine wave
    carrier_freq = 100  # Carrier frequency (Hz)
    mod_freq = 2  # Modulation frequency (Hz)
    freq_deviation = 50  # Frequency deviation (Hz)
    
    # Instantaneous frequency varies as: f(t) = f_c + Δf * sin(2π * f_m * t)
    instantaneous_phase = 2 * np.pi * carrier_freq * t + (freq_deviation / mod_freq) * np.cos(2 * np.pi * mod_freq * t)
    signal_data = np.sin(instantaneous_phase)
    
    # Add a chirp component for more complexity (frequency sweeps from 200 to 50 Hz)
    chirp = signal.chirp(t, f0=200, f1=50, t1=duration, method='linear')
    
    # Combine FM and chirp with a brief pulse
    pulse = np.exp(-((t - 1.5) / 0.3) ** 2) * np.sin(2 * np.pi * 150 * t)
    
    signal_data = 0.5 * signal_data + 0.3 * chirp + 0.4 * pulse

    # Compute and plot the spectrogram
    spectogram(signal_data, fs, nSegment=256, overlapPorcentage=75, window='hamming')