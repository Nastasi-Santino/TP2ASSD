import numpy as np
import soundfile as sf

def simple_lowpass_filter(x, cutoff_norm):
    """
    Simple moving average lowpass filter.
    
    Parameters:
    x: input signal
    cutoff_norm: normalized cutoff frequency (0 to 1)
    
    Returns:
    y: filtered signal
    """
    # Calculate filter kernel size based on cutoff frequency
    kernel_size = max(3, int(1.0 / cutoff_norm))
    kernel = np.ones(kernel_size) / kernel_size
    
    # Apply filter with padding to preserve length
    y = np.convolve(x, kernel, mode='same')
    return y

def cubic_interpolate(x_data, y_data, x_new):
    """
    Cubic Hermite spline interpolation.
    
    Parameters:
    x_data: original x coordinates
    y_data: original y values
    x_new: new x coordinates to interpolate at
    
    Returns:
    y_new: interpolated values
    """
    y_new = np.zeros_like(x_new, dtype=float)
    
    for i, xi in enumerate(x_new):
        # Find the two points to interpolate between
        idx = int(np.floor(xi))
        
        # Clamp to valid range
        if idx < 0:
            y_new[i] = y_data[0]
            continue
        if idx >= len(y_data) - 1:
            y_new[i] = y_data[-1]
            continue
        
        # Get the four surrounding points for cubic interpolation
        x0 = max(0, idx - 1)
        x1 = idx
        x2 = min(len(y_data) - 1, idx + 1)
        x3 = min(len(y_data) - 1, idx + 2)
        
        y0 = y_data[x0]
        y1 = y_data[x1]
        y2 = y_data[x2]
        y3 = y_data[x3]
        
        # Local position in segment (0 to 1)
        t = xi - idx
        
        # Catmull-Rom interpolation formula
        a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        a2 = -0.5 * y0 + 0.5 * y2
        a3 = y1
        
        y_new[i] = a0 * t**3 + a1 * t**2 + a2 * t + a3
    
    return y_new

def PSOLA_time_stretch(x, pitchMarks, alpha):
    """
    PSOLA time stretch algorithm with pitch preservation.
    
    Time-stretches audio without changing pitch. Maintains the original pitch period
    by cycling through input segments and replicating them to fill the stretched duration.
    Uses Hann windowing and overlap-and-add reconstruction.
    
    Parameters:
    x: input signal
    pitchMarks: array of pitch mark indices
    alpha: time stretch factor (alpha > 1 for stretching, alpha < 1 for compressing)
    
    Returns:
    y: time-stretched signal with preserved pitch
    """
    
    # Convert pitch marks to integers
    pitchMarks = np.asarray(pitchMarks, dtype=int)

    # Estimate the median pitch period from the input signal
    T0 = int(np.median(np.diff(pitchMarks)))
    
    # Define the half-length of the window (typically equal to one pitch period)
    half_len = T0
    win_len = 2 * half_len + 1

    # Create Hann window centered at each pitch mark
    window = np.hanning(win_len)

    # Allocate output buffer with extra padding to avoid boundary issues
    y_len = int(len(x) * alpha) + 2 * half_len
    y = np.zeros(y_len)
    norm = np.zeros(y_len)

    # Calculate output pitch marks with original spacing (no pitch change)
    # Create output marks at intervals of T0 to fill the stretched duration
    num_output_marks = int((len(x) * alpha) / T0)
    new_pitch_marks = np.arange(num_output_marks) * T0

    # Process each output pitch mark by cycling through input segments
    # This repeats input periods to maintain pitch while stretching duration
    for i, pm_out in enumerate(new_pitch_marks):
        # Map output position back to input position proportionally (no abrupt wrapping)
        # This creates smooth continuous progression through input segments
        input_position = (i / num_output_marks) * (len(pitchMarks) - 1)
        pm_in_idx = int(input_position)
        
        # Clamp to valid range
        if pm_in_idx >= len(pitchMarks) - 1:
            pm_in_idx = len(pitchMarks) - 2
        
        pm_in = pitchMarks[pm_in_idx]

        # Define the segment boundaries centered at the input pitch mark
        in_start = pm_in - half_len
        in_end = pm_in + half_len + 1

        # Define the output boundaries centered at the output pitch mark (same spacing)
        out_start = pm_out - half_len
        out_end = pm_out + half_len + 1

        # Skip if segment extends beyond input or output boundaries
        if in_start < 0 or in_end > len(x):
            continue
        if out_start < 0 or out_end > len(y):
            continue

        # Extract windowed segment from input
        segment = x[in_start:in_end] * window

        # Add to output using overlap-and-add reconstruction
        y[out_start:out_end] += segment
        norm[out_start:out_end] += window

    # Normalize by the window overlap to maintain amplitude consistency
    valid = norm > 1e-8
    y[valid] /= norm[valid]

    return y

def PSOLA_pitch_shift(x, pitchMarks, beta):
    """
    PSOLA pitch shift algorithm with anti-aliasing and cubic interpolation.
    
    Shifts the pitch of audio while maintaining duration. Works by time-stretching
    inversely and then resampling back to the original length.
    
    Parameters:
    x: input signal
    pitchMarks: array of pitch mark indices
    beta: pitch shift factor (beta > 1 for higher pitch, beta < 1 for lower pitch)
    
    Returns:
    y: pitch-shifted audio with original duration
    """
    
    # Time-stretch by beta to change pitch without changing duration
    # e.g., beta=2 (pitch up octave) → stretch by 2 → then resample down to original length
    temp = PSOLA_time_stretch(x, pitchMarks, beta)

    # Apply anti-aliasing filter when beta > 1 (downsampling case)
    # This prevents high-frequency aliasing artifacts when resampling down
    if beta > 1:
        # Apply simple lowpass filter at normalized cutoff frequency 1/beta
        temp = simple_lowpass_filter(temp, 1.0 / beta)

    # Target output length (same as input to preserve duration)
    y_len = len(x)

    # Resample using cubic Hermite spline interpolation for smoother, higher-quality results
    x_new = np.linspace(0, len(temp) - 1, y_len)
    y = cubic_interpolate(np.arange(len(temp)), temp, x_new)

    return y
# Test bench

if __name__ == "__main__":
    # Load the cropped audio
    audio, sr = sf.read('PitchMarksPSOLA/CelloC32_cropped.wav')

    # Load pitch mark indices
    marks = np.load('PitchMarksPSOLA/CelloC32_pitch_marks_indices.npy')

    # Define the scale from C3 to C5 (original is C4)
    # Semitone offsets from C4 for each note
    scale_notes = [
        ('C3', -12),  # Down 1 octave
        ('D3', -10),
        ('E3', -8),
        ('F3', -7),
        ('G3', -5),
        ('A3', -3),
        ('B3', -1),
        ('C4', 0),    # Original note
        ('D4', 2),
        ('E4', 4),
        ('F4', 5),
        ('G4', 7),
        ('A4', 9),
        ('B4', 11),
        ('C5', 12),   # Up 1 octave
    ]
    
    # Generate pitch-shifted notes and concatenate
    scale_audio = []
    
    for note_name, semitones in scale_notes:
        # Calculate pitch shift factor from semitone offset
        # beta = 2^(semitones/12)
        beta = 2 ** (semitones / 12)
        
        # Pitch shift the audio
        shifted = PSOLA_pitch_shift(audio, marks, beta)
        scale_audio.append(shifted)
        
        print(f"Generated {note_name} (semitones: {semitones:+d}, beta: {beta:.4f})")
    
    # Concatenate all notes into a single audio sequence
    full_scale = np.concatenate(scale_audio)
    
    # Save the scale
    sf.write('CelloScale_C32_to_C5.wav', full_scale, sr)
    print("Scale saved as 'CelloScale_C32_to_C5.wav'")