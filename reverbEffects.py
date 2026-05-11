import numpy as np
from scipy.io import wavfile
from scipy import signal

def simple_eco(x, delay_time, decay_factor, fs=44100):
    """
    Simple echo effect.

    Implements: y[n] = x[n] + g * x[n - D]
    
    Parameters:
    x: input signal
    delay_time: delay time in seconds
    decay_factor: decay factor for the echo (0 < decay_factor < 1)
    fs: sampling rate (default 44100 Hz)
    
    Returns:
    y: output signal with echo effect
    """
    delay_samples = int(delay_time * fs)
    y = np.copy(x)
    
    for n in range(delay_samples, len(x)):
        y[n] += decay_factor * x[n - delay_samples]
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y /= max_val
    
    return y

def flat_reverb(x, delay_time, gain, fs=44100):
    """
    Planar reverberator using real-time feedback.
    
    Implements: y[n] = x[n] + g * y[n - D]
    
    Parameters:
    x: input signal
    delay_time: delay time D in seconds
    gain: feedback gain coefficient g (0 < g < 1 for stability)
    fs: sampling rate (default 44100 Hz)
    
    Returns:
    y: output signal with reverb effect
    """
    delay_samples = int(delay_time * fs)
    y = np.zeros(len(x))
    
    for n in range(len(x)):
        if n >= delay_samples:
            y[n] = x[n] + gain * y[n - delay_samples]
        else:
            y[n] = x[n]
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y /= max_val
    
    return y

def low_pass_reverb(x, delay_time, gain, damping, fs=44100):
    """
    Low-pass reverberator with high-frequency absorption.
    
    In real environments, high frequencies are absorbed faster.
    This reverberator adds a low-pass filter to the feedback path.
    
    Implements: y[n] = x[n] + g((1-a)y[n-D] + ay[n-D-1])
    
    Parameters:
    x: input signal
    delay_time: delay time D in seconds (temporal density of reflections)
    gain: feedback gain coefficient g (controls reverb duration, 0 < g < 1)
    damping: high-frequency attenuation coefficient a (0 to 1)
             a → 0: almost flat response
             a → 1: darker, more dampened sound
    fs: sampling rate (default 44100 Hz)
    
    Returns:
    y: output signal with low-pass filtered reverb effect
    """
    delay_samples = int(delay_time * fs)
    y = np.zeros(len(x))
    
    for n in range(len(x)):
        if n >= delay_samples + 1:
            # Apply low-pass filter to feedback: (1-a)y[n-D] + a*y[n-D-1]
            filtered_feedback = (1 - damping) * y[n - delay_samples] + damping * y[n - delay_samples - 1]
            y[n] = x[n] + gain * filtered_feedback
        elif n >= delay_samples:
            # Handle boundary case where n-D-1 would be negative
            y[n] = x[n] + gain * (1 - damping) * y[n - delay_samples]
        else:
            y[n] = x[n]
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y /= max_val
    
    return y

def convolution_reverb(x, preset='church'):
    """
    Convolution reverb using impulse responses.
    
    Convolution with an impulse response simulates the reverberation of a specific space.
    
    Parameters:
    x: input signal
    preset: reverb preset to use (e.g., 'church', 'hall', 'room')
    
    Returns:
    y: output signal convolved with the impulse response
    """
    # Load impulse response based on preset
    if preset == 'church':
        fs_ir, ir = wavfile.read('ImpulseResponses/1st_Baptist_Nashville.wav')
    elif preset == 'hall1':
        fs_ir, ir = wavfile.read('ImpulseResponses/Central_Hall_University_York.wav')
    elif preset == 'hall2':
        fs_ir, ir = wavfile.read('ImpulseResponses/Elveden_Hall.wav')
    elif preset == 'nuclearReactor':
        fs_ir, ir = wavfile.read('ImpulseResponses/Nuclear_Reactor.wav')
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    # Convert to float and normalize if necessary
    if ir.dtype != np.float32 and ir.dtype != np.float64:
        ir = ir.astype(np.float32) / np.iinfo(ir.dtype).max
    
    # Convolve input signal with impulse response
    y = signal.fftconvolve(x, ir, mode='full')
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y /= max_val
    
    return y
