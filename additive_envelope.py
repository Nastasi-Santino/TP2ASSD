from scipy.signal import butter, filtfilt, hilbert
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def envelope_rect_lpf(x, fs, fc=20):
    # Rectificación
    x_abs = np.abs(x)

    # Filtro pasa bajos
    b, a = butter(4, fc/(fs/2), btype='low')

    # Filtrado sin desfase
    env = filtfilt(b, a, x_abs)

    # Normalización
    env = env / np.max(env)

    return env


def envelope_hilbert(x):
    analytic_signal = hilbert(x)
    env = np.abs(analytic_signal)

    env = env / np.max(env)

    return env

# Test Bench
if __name__ == "__main__":
    # ============ SELECT AUDIO FILE ============
    audio_file = "FluteC4.wav"  # Change this to select different instrument
    # ============================================
    
    # Load audio
    filepath = f"InstrumentsPSOLA/{audio_file}"
    sr, audio = wavfile.read(filepath)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Extract envelopes
    env1 = envelope_rect_lpf(audio, sr, fc=20)
    env2 = envelope_hilbert(audio)

    # Time array
    time = np.arange(len(audio)) / sr
    
    threshold_attack = 0.75 * np.max(env1)
    i_A = np.where(env1 >= threshold_attack)[0][0]
    A_time = time[i_A]
    A_fraction  = A_time / (len(audio) / sr)

    threshold_release_start = 0.85 * np.max(env1)
    # buscar cerca del final dónde deja de estar alta
    i_R_start = np.where((time > 1.8) & (env1 < threshold_release_start))[0][0]
    R_fraction = (len(audio) / sr - time[i_R_start]) / (len(audio) / sr)

    t_s = time[i_A:i_R_start]
    env_s = env1[i_A:i_R_start]
    coeffs = np.polyfit(t_s, env_s, 1)

    alpha = coeffs[0]
    A0 = coeffs[1]

    print(f"Attack time: {A_time:.3f} s ({A_fraction:.2%} of total duration)")
    print(f"Release start time: {time[i_R_start]:.3f} s ({R_fraction:.2%} of total duration)")
    print(f"Linear fit coefficients: alpha = {alpha:.4f}, A0 = {A0:.4f}")

    # Plot
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Waveform + Rect+LPF envelope
    plt.subplot(2, 1, 1)
    #plt.plot(time, audio, linewidth=0.5, color='steelblue', alpha=0.7, label='Waveform')
    plt.plot(time, env1, linewidth=2, color='red', label='Rect + LPF')
    #plt.plot(time, -env1, linewidth=2, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Rect + LPF Envelope: {audio_file}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Waveform + Hilbert envelope
    plt.subplot(2, 1, 2)
    #plt.plot(time, audio, linewidth=0.5, color='steelblue', alpha=0.7, label='Waveform')
    plt.plot(time, env2, linewidth=2, color='green', label='Hilbert')
    #plt.plot(time, -env2, linewidth=2, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Hilbert Envelope: {audio_file}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'envelope_{audio_file.replace(".wav", ".png")}', dpi=150)
    plt.show()
    