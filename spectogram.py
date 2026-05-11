import scipy.signal as signal
import scipy.io.wavfile as wavfile
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
    # Load the G3 major scale audio file
    fs, audio = wavfile.read('G3_major_scale.wav')
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    print(f"Loaded G3_major_scale.wav")
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(audio) / fs:.3f} seconds")
    print(f"\nGenerating spectrogram...")
    
    # Compute and plot the spectrogram
    spectogram(audio, fs, nSegment=1024, overlapPorcentage=75, window='hann')