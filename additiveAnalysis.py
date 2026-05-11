import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
from pathlib import Path

# ============================================================================
# ADDITIVE SYNTHESIS ANALYSIS - FFT AND HARMONICS EXTRACTION
# ============================================================================

class AdditiveAnalyzer:
    """Analyzes audio signals for additive synthesis using FFT and harmonic extraction"""
    
    def __init__(self, audio_folder="PitchMarksPSOLA"):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        audio_folder : str
            Path to folder containing cropped .wav files
        """
        self.audio_folder = audio_folder
        self.wav_files = []
        self.audio_data = {}
        self.fft_data = {}
        self.harmonics = {}
        
    def load_wav_files(self):
        """Load all .wav files from the specified folder"""
        print(f"Loading .wav files from {self.audio_folder}...")
        
        for file in os.listdir(self.audio_folder):
            if file.endswith(".wav"):
                filepath = os.path.join(self.audio_folder, file)
                try:
                    sr, audio = wavfile.read(filepath)
                    self.wav_files.append(file)
                    self.audio_data[file] = {'sample_rate': sr, 'audio': audio}
                    print(f"  ✓ Loaded {file} (Sample Rate: {sr} Hz, Duration: {len(audio)/sr:.3f}s)")
                except Exception as e:
                    print(f"  ✗ Error loading {file}: {e}")
        
        if not self.wav_files:
            print("No .wav files found!")
            return False
        return True
    
    def compute_fft(self, audio, sr):
        """
        Compute FFT of audio signal
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio signal (mono or first channel if stereo)
        sr : int
            Sample rate
        
        Returns:
        --------
        freqs : np.ndarray
            Frequency values
        magnitude : np.ndarray
            Magnitude spectrum (normalized)
        """
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Apply Hann window to reduce spectral leakage
        window = np.hanning(len(audio))
        windowed_audio = audio * window
        
        # Compute FFT
        fft_result = fftpack.fft(windowed_audio)
        magnitude = np.abs(fft_result)
        
        # Only keep positive frequencies
        n = len(magnitude)
        magnitude = magnitude[:n//2]
        freqs = fftpack.fftfreq(n, 1/sr)[:n//2]
        
        # Normalize magnitude
        magnitude = magnitude / np.max(magnitude)
        
        return freqs, magnitude
    
    def find_harmonics(self, freqs, magnitude, prominence_threshold=0.05, max_harmonics=20):
        """
        Find harmonic peaks in the frequency spectrum
        
        Parameters:
        -----------
        freqs : np.ndarray
            Frequency array
        magnitude : np.ndarray
            Magnitude spectrum
        prominence_threshold : float
            Minimum prominence to detect peaks (0-1)
        max_harmonics : int
            Maximum number of harmonics to extract
        
        Returns:
        --------
        harmonics_dict : dict
            Dictionary with harmonic information {harmonic_number: {'freq': freq, 'amp': amplitude}}
        fundamental_freq : float
            Detected fundamental frequency
        """
        # Find peaks with prominence threshold
        peaks, properties = find_peaks(magnitude, prominence=prominence_threshold, distance=5)
        
        # Sort peaks by magnitude (highest first)
        peak_indices = peaks[np.argsort(-magnitude[peaks])][:max_harmonics]
        peak_indices = np.sort(peak_indices)
        
        if len(peak_indices) == 0:
            print("No peaks found!")
            return {}, 0
        
        # The first (lowest frequency) peak should be the fundamental
        fundamental_idx = peak_indices[0]
        fundamental_freq = freqs[fundamental_idx]
        fundamental_amp = magnitude[fundamental_idx]
        
        # Extract harmonics with amplitudes relative to fundamental
        harmonics_dict = {}
        for i, idx in enumerate(peak_indices):
            harmonic_freq = freqs[idx]
            harmonic_amp = magnitude[idx]
            
            # Calculate harmonic number relative to fundamental
            if fundamental_freq > 0:
                harmonic_number = round(harmonic_freq / fundamental_freq)
            else:
                harmonic_number = i + 1
            
            # Amplitude relative to first harmonic (in dB)
            if fundamental_amp > 0:
                amp_relative = 20 * np.log10(harmonic_amp / fundamental_amp)
            else:
                amp_relative = 0
            
            harmonics_dict[harmonic_number] = {
                'freq': harmonic_freq,
                'amp_linear': harmonic_amp / fundamental_amp if fundamental_amp > 0 else 0,
                'amp_relative_dB': amp_relative
            }
        
        return harmonics_dict, fundamental_freq
    
    def analyze_all_files(self, prominence=0.05):
        """Analyze all loaded audio files"""
        print("\n" + "="*70)
        print("ANALYZING FFT AND HARMONICS")
        print("="*70)
        
        for wav_file in self.wav_files:
            print(f"\nAnalyzing: {wav_file}")
            print("-" * 70)
            
            sr = self.audio_data[wav_file]['sample_rate']
            audio = self.audio_data[wav_file]['audio']
            
            # Compute FFT
            freqs, magnitude = self.compute_fft(audio, sr)
            self.fft_data[wav_file] = {'freqs': freqs, 'magnitude': magnitude}
            
            # Find harmonics
            harmonics, fundamental = self.find_harmonics(freqs, magnitude, prominence_threshold=prominence)
            self.harmonics[wav_file] = {'harmonics': harmonics, 'fundamental': fundamental}
            
            # Print results
            print(f"Fundamental frequency: {fundamental:.2f} Hz")
            print(f"Number of harmonics detected: {len(harmonics)}")
            print("\nHarmonic Content:")
            print(f"{'Harmonic':<12} {'Frequency (Hz)':<18} {'Amplitude (linear)':<22} {'Amplitude (dB)':<16}")
            print("-" * 70)
            
            for h_num in sorted(harmonics.keys()):
                h_info = harmonics[h_num]
                print(f"{h_num:<12} {h_info['freq']:<18.2f} {h_info['amp_linear']:<22.4f} {h_info['amp_relative_dB']:<16.2f}")
    
    def plot_fft(self, freq_range_hz=None):
        """
        Plot FFT for all files
        
        Parameters:
        -----------
        freq_range_hz : tuple
            Optional (min_freq, max_freq) to limit x-axis
        """
        n_files = len(self.wav_files)
        fig, axes = plt.subplots(n_files, 1, figsize=(14, 5*n_files))
        
        if n_files == 1:
            axes = [axes]
        
        for idx, wav_file in enumerate(self.wav_files):
            if wav_file not in self.fft_data:
                continue
            
            freqs = self.fft_data[wav_file]['freqs']
            magnitude = self.fft_data[wav_file]['magnitude']
            harmonics = self.harmonics[wav_file]['harmonics']
            fundamental = self.harmonics[wav_file]['fundamental']
            
            ax = axes[idx]
            
            # Plot FFT
            ax.plot(freqs, magnitude, 'b-', linewidth=0.8, label='FFT Magnitude')
            ax.grid(True, alpha=0.3)
            
            # Mark harmonics
            harmonic_freqs = [h_info['freq'] for h_info in harmonics.values()]
            harmonic_mags = [h_info['amp_linear'] for h_info in harmonics.values()]
            ax.scatter(harmonic_freqs, harmonic_mags, color='red', s=50, zorder=5, label='Detected Harmonics')
            
            # Mark fundamental
            if fundamental > 0:
                fundamental_mag = magnitude[np.argmin(np.abs(freqs - fundamental))]
                ax.axvline(fundamental, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Fundamental ({fundamental:.2f} Hz)')
            
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Magnitude (normalized)', fontsize=11)
            ax.set_title(f'FFT Analysis: {wav_file}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            
            if freq_range_hz:
                ax.set_xlim(freq_range_hz)
            else:
                # Auto-limit to reasonable frequency range
                ax.set_xlim(0, max(5000, fundamental * 10) if fundamental > 0 else 5000)
        
        plt.tight_layout()
        plt.savefig('fft_analysis.png', dpi=150, bbox_inches='tight')
        print("\n✓ FFT plot saved as 'fft_analysis.png'")
        plt.show()
    
    def plot_harmonics_bar(self):
        """Plot harmonic amplitudes as bar charts"""
        n_files = len(self.wav_files)
        fig, axes = plt.subplots(n_files, 1, figsize=(12, 5*n_files))
        
        if n_files == 1:
            axes = [axes]
        
        for idx, wav_file in enumerate(self.wav_files):
            if wav_file not in self.harmonics:
                continue
            
            harmonics = self.harmonics[wav_file]['harmonics']
            
            ax = axes[idx]
            
            harmonic_nums = sorted(harmonics.keys())
            amplitudes_dB = [harmonics[h]['amp_relative_dB'] for h in harmonic_nums]
            
            colors = ['green' if h == 1 else 'blue' for h in harmonic_nums]
            ax.bar(harmonic_nums, amplitudes_dB, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Harmonic Number', fontsize=11)
            ax.set_ylabel('Amplitude (dB re. Fundamental)', fontsize=11)
            ax.set_title(f'Harmonic Content: {wav_file}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            
            # Set x-axis to show all harmonics
            ax.set_xticks(harmonic_nums)
        
        plt.tight_layout()
        plt.savefig('harmonics_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Harmonics bar chart saved as 'harmonics_analysis.png'")
        plt.show()
    
    def plot_waveforms(self):
        """Plot the time-domain waveforms"""
        n_files = len(self.wav_files)
        fig, axes = plt.subplots(n_files, 1, figsize=(14, 4*n_files))
        
        if n_files == 1:
            axes = [axes]
        
        for idx, wav_file in enumerate(self.wav_files):
            sr = self.audio_data[wav_file]['sample_rate']
            audio = self.audio_data[wav_file]['audio']
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            ax = axes[idx]
            
            # Time axis
            time = np.arange(len(audio)) / sr
            
            ax.plot(time, audio, linewidth=0.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.set_title(f'Waveform: {wav_file}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('waveforms.png', dpi=150, bbox_inches='tight')
        print("✓ Waveforms plot saved as 'waveforms.png'")
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def select_audio_file(analyzer):
    """
    Display available audio files and let user select one
    
    Returns:
    --------
    str : selected filename or None if invalid selection
    """
    if not analyzer.wav_files:
        print("No audio files available!")
        return None
    
    print("\n" + "="*70)
    print("AVAILABLE AUDIO FILES")
    print("="*70)
    
    for idx, filename in enumerate(analyzer.wav_files, 1):
        sr = analyzer.audio_data[filename]['sample_rate']
        audio = analyzer.audio_data[filename]['audio']
        duration = len(audio) / sr
        print(f"{idx}. {filename:<30} (SR: {sr} Hz, Duration: {duration:.3f}s)")
    
    print("\n" + "-"*70)
    while True:
        try:
            choice = int(input(f"Select audio file (1-{len(analyzer.wav_files)}): "))
            if 1 <= choice <= len(analyzer.wav_files):
                selected = analyzer.wav_files[choice - 1]
                print(f"✓ Selected: {selected}")
                return selected
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(analyzer.wav_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def analyze_single_file(analyzer, filename, prominence=0.03):
    """Analyze a single file and display results"""
    
    print("\n" + "="*70)
    print(f"ANALYZING: {filename}")
    print("="*70)
    
    sr = analyzer.audio_data[filename]['sample_rate']
    audio = analyzer.audio_data[filename]['audio']
    
    # Compute FFT
    freqs, magnitude = analyzer.compute_fft(audio, sr)
    analyzer.fft_data[filename] = {'freqs': freqs, 'magnitude': magnitude}
    
    # Find harmonics
    harmonics, fundamental = analyzer.find_harmonics(freqs, magnitude, prominence_threshold=prominence)
    analyzer.harmonics[filename] = {'harmonics': harmonics, 'fundamental': fundamental}
    
    # Print results
    print(f"\nFundamental frequency: {fundamental:.2f} Hz")
    print(f"Number of harmonics detected: {len(harmonics)}")
    print("\nHarmonic Content:")
    print(f"{'Harmonic':<12} {'Frequency (Hz)':<18} {'Amplitude (linear)':<22} {'Amplitude (dB)':<16}")
    print("-" * 70)
    
    for h_num in sorted(harmonics.keys()):
        h_info = harmonics[h_num]
        print(f"{h_num:<12} {h_info['freq']:<18.2f} {h_info['amp_linear']:<22.4f} {h_info['amp_relative_dB']:<16.2f}")


def plot_single_file(analyzer, filename, freq_range_hz=None):
    """Plot results for a single file"""
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 12))
    
    sr = analyzer.audio_data[filename]['sample_rate']
    audio = analyzer.audio_data[filename]['audio']
    freqs = analyzer.fft_data[filename]['freqs']
    magnitude = analyzer.fft_data[filename]['magnitude']
    harmonics = analyzer.harmonics[filename]['harmonics']
    fundamental = analyzer.harmonics[filename]['fundamental']
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # ---- Plot 1: Waveform ----
    ax1 = plt.subplot(3, 1, 1)
    time = np.arange(len(audio)) / sr
    ax1.plot(time, audio, linewidth=0.8, color='steelblue')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Waveform: {filename}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ---- Plot 2: FFT ----
    ax2 = plt.subplot(3, 1, 2)
    
    # Normalize magnitude to fundamental
    fundamental_idx = np.argmin(np.abs(freqs - fundamental))
    fundamental_mag = magnitude[fundamental_idx]
    if fundamental_mag > 0:
        magnitude_normalized = magnitude / fundamental_mag
    else:
        magnitude_normalized = magnitude
    
    ax2.plot(freqs, magnitude_normalized, 'b-', linewidth=0.8, label='FFT Magnitude')
    
    # Mark harmonics
    harmonic_freqs = [h_info['freq'] for h_info in harmonics.values()]
    harmonic_mags = [h_info['amp_linear'] for h_info in harmonics.values()]
    ax2.scatter(harmonic_freqs, harmonic_mags, color='red', s=80, zorder=5, label='Detected Harmonics', edgecolors='darkred')
    
    # Mark fundamental
    if fundamental > 0:
        fundamental_mag = magnitude[np.argmin(np.abs(freqs - fundamental))]
        ax2.axvline(fundamental, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Fundamental ({fundamental:.2f} Hz)')
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Magnitude (normalized)', fontsize=11)
    ax2.set_title(f'FFT Analysis: {filename}', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    if freq_range_hz:
        ax2.set_xlim(freq_range_hz)
    else:
        ax2.set_xlim(0, max(5000, fundamental * 10) if fundamental > 0 else 5000)
    
    # ---- Plot 3: Harmonics Bar Chart ----
    ax3 = plt.subplot(3, 1, 3)
    harmonic_nums = sorted(harmonics.keys())
    amplitudes_dB = [harmonics[h]['amp_relative_dB'] for h in harmonic_nums]
    
    colors = ['green' if h == 1 else 'steelblue' for h in harmonic_nums]
    ax3.bar(harmonic_nums, amplitudes_dB, color=colors, alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('Harmonic Number', fontsize=11)
    ax3.set_ylabel('Amplitude (dB re. Fundamental)', fontsize=11)
    ax3.set_title(f'Harmonic Content: {filename}', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xticks(harmonic_nums)
    
    plt.tight_layout()
    
    # Save with filename
    save_name = f"analysis_{filename.replace('.wav', '.png')}"
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Analysis plot saved as '{save_name}'")
    plt.show()


if __name__ == "__main__":
    # Create analyzer
    analyzer = AdditiveAnalyzer(audio_folder="PitchMarksPSOLA")
    
    # Load audio files
    if not analyzer.load_wav_files():
        print("Failed to load audio files. Exiting.")
        exit(1)
    
    # Select audio file to analyze
    selected_file = select_audio_file(analyzer)
    
    if selected_file is None:
        print("No file selected. Exiting.")
        exit(1)
    
    # Analyze the selected file
    analyze_single_file(analyzer, selected_file, prominence=0.03)
    
    # Plot the results
    plot_single_file(analyzer, selected_file, freq_range_hz=(0, 4000))
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
