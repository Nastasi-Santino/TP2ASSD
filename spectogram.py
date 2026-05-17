import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np


def spectogram(data, fs, nSegment=256, overlapPorcentage=None, window='hann'):

    # Si no especifica, default 0% de overlap
    if overlapPorcentage is None:
        noverlap = 0
    else:
        noverlap = int(nSegment * overlapPorcentage / 100)

    # Distancia entre segmentos solapados
    stride = nSegment - noverlap
    
    # Genera indices de inicio de cada segmento
    segment_starts = range(0, len(data) - nSegment + 1, stride)
    
    # Extrae segmentos de la señal
    segments = np.array([data[i:i + nSegment] for i in segment_starts])

    # Elige la ventana y aplica la FFT a cada segmento
    try:
        window_func = signal.get_window(window, nSegment)
    except ValueError:
        raise ValueError("Invalid window specification")

    ffts = np.fft.fft(segments * window_func, axis=1)
    
    # Solo nos interesan las frecuencias positivas y la magnitud
    magnitude = np.abs(ffts)
    magnitude = magnitude[:, :nSegment // 2]  # Solo la mitad positiva
    
    # Eje de frecuencias (Hz)
    freqs = np.fft.fftfreq(nSegment, 1/fs)[:nSegment // 2]
    
    # Eje de tiempo (segundos) - centro de cada segmento
    times = np.array([i / fs for i in segment_starts])
    
    # Graficar el espectrograma
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

    # Carga la escala de sol mayor
    fs, audio = wavfile.read('G3_major_scale.wav')
    
    # Convierte a mono si es estéreo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    print(f"Loaded G3_major_scale.wav")
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(audio) / fs:.3f} seconds")
    print(f"\nGenerating spectrogram...")
    
    # Genera el espectrograma con 1024 muestras por segmento y 75% de overlap
    spectogram(audio, fs, nSegment=1024, overlapPorcentage=75, window='hann')