"""
PSOLA Pitch Detector - Musical Synthesizer Implementation
Detects pitch marks and fundamental frequency from monophonic instruments
"""

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import librosa
    import librosa.display
    USE_LIBROSA = True
except ImportError:
    import scipy.io.wavfile as wavfile
    USE_LIBROSA = False

# ==================== MACRO: SELECT INSTRUMENT ====================
# Change this to select which instrument to analyze
# Available options: 'FluteC4', 'GuitarC4', 'PianoC4', etc.
INSTRUMENT = 'PianoC3'
# ===================================================================


class PSLAPitchDetector:
    """
    Pitch detector for monophonic instruments using autocorrelation method.
    Detects fundamental frequency and pitch marks for PSOLA synthesis.
    """
    
    def __init__(self, wav_file_path, min_freq=50, max_freq=2000, save_plots=True, show_plots=False):
        """
        Initialize the pitch detector.
        
        Parameters:
        - wav_file_path: Path to the .wav file
        - min_freq: Minimum frequency to search for (Hz)
        - max_freq: Maximum frequency to search for (Hz)
        - save_plots: Whether to save plots to files
        - show_plots: Whether to display plots (not recommended for batch processing)
        """
        self.wav_path = wav_file_path
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.fs = None
        self.signal = None
        self.duration = None
        
    def load_audio(self):
        """Load audio file."""
        try:
            if USE_LIBROSA:
                self.signal, self.fs = librosa.load(str(self.wav_path), sr=None, mono=True)
            else:
                self.fs, audio_data = wavfile.read(self.wav_path)
                self.signal = audio_data.astype(float)
                # Convert stereo to mono if necessary
                if len(self.signal.shape) > 1:
                    self.signal = np.mean(self.signal, axis=1)
                # Normalize
                self.signal = self.signal / np.max(np.abs(self.signal))
            
            self.duration = len(self.signal) / self.fs
            print(f"✓ Loaded: {self.wav_path}")
            print(f"  Sampling frequency: {self.fs} Hz")
            print(f"  Duration: {self.duration:.2f} seconds")
            print(f"  Signal shape: {self.signal.shape}")
            return True
        except FileNotFoundError:
            print(f"✗ File not found: {self.wav_path}")
            return False
        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            return False
    
    def compute_spectrogram(self, nSegment=512, overlap_percentage=75, window='hann'):
        """
        Compute and display spectrogram.
        
        Parameters:
        - nSegment: Length of each segment
        - overlap_percentage: Overlap between segments (%)
        - window: Window function to use
        """
        if self.signal is None:
            print("Signal not loaded. Call load_audio() first.")
            return
        
        noverlap = int(nSegment * overlap_percentage / 100)
        stride = nSegment - noverlap
        
        # Generate segment starting indices
        segment_starts = range(0, len(self.signal) - nSegment + 1, stride)
        
        # Extract overlapping segments
        segments = np.array([self.signal[i:i + nSegment] for i in segment_starts])
        
        # Apply window
        window_func = signal.get_window(window, nSegment)
        ffts = np.fft.fft(segments * window_func, axis=1)
        
        # Compute magnitude spectrum
        magnitude = np.abs(ffts)
        magnitude = magnitude[:, :nSegment // 2]
        
        # Frequency and time axes
        freqs = np.fft.fftfreq(nSegment, 1/self.fs)[:nSegment // 2]
        times = np.array([i / self.fs for i in segment_starts])
        
        # Plot spectrogram
        plt.figure(figsize=(14, 6))
        plt.pcolormesh(times, freqs, 20 * np.log10(magnitude.T + 1e-10), 
                       shading='auto', cmap='viridis')
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Spectrogram - {INSTRUMENT}')
        plt.ylim([0, 1500])  # Focus on typical instrument range
        plt.tight_layout()
        
        if self.save_plots:
            output_file = Path(self.wav_path).parent.parent / f'{INSTRUMENT}_spectrogram.png'
            plt.savefig(str(output_file), dpi=100, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Spectrogram computed (segments: {len(segments)})")
    
    def detect_fundamental_frequency(self, frame_length=4096, hop_length=512):
        """
        Detect fundamental frequency using autocorrelation method.
        This method is well-suited for PSOLA synthesis.
        
        Parameters:
        - frame_length: Length of analysis frame
        - hop_length: Hop size between frames
        
        Returns:
        - times: Time of each frame
        - frequencies: Fundamental frequency for each frame
        """
        if self.signal is None:
            print("Signal not loaded. Call load_audio() first.")
            return None, None
        
        frequencies = []
        times = []
        
        num_frames = 1 + (len(self.signal) - frame_length) // hop_length
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = self.signal[start:end]
            
            if len(frame) < frame_length:
                break
            
            # Apply window
            frame = frame * signal.windows.hann(len(frame))
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find lag range corresponding to min/max frequencies
            lag_min = int(self.fs / self.max_freq)
            lag_max = int(self.fs / self.min_freq)
            
            if lag_max < len(autocorr):
                # Find peak in valid range
                peak_lag = np.argmax(autocorr[lag_min:lag_max]) + lag_min
                fundamental_freq = self.fs / peak_lag
                frequencies.append(fundamental_freq)
                times.append((start + frame_length // 2) / self.fs)
        
        frequencies = np.array(frequencies)
        times = np.array(times)
        
        print(f"✓ Detected fundamental frequency")
        print(f"  Mean frequency: {np.mean(frequencies):.2f} Hz")
        print(f"  Frequency range: {np.min(frequencies):.2f} - {np.max(frequencies):.2f} Hz")
        
        return times, frequencies
    
    def detect_pitch_marks(self, times, frequencies, threshold_percentile=70, min_distance_ms=20):
        """
        Detect pitch marks (one per pitch period for PSOLA).
        Places marks at maximum amplitude within each pitch period.
        
        Parameters:
        - times: Time array from fundamental frequency detection
        - frequencies: Fundamental frequency array
        - threshold_percentile: Percentile threshold for voiced/unvoiced decision
        - min_distance_ms: Minimum distance between pitch marks (unused, kept for compatibility)
        
        Returns:
        - pitch_marks: Time positions of all detected pitch marks
        - pitch_mark_frequencies: F0 at each pitch mark
        """
        if times is None or frequencies is None:
            print("Fundamental frequency not detected. Call detect_fundamental_frequency() first.")
            return None, None
        
        # Compute energy in each frame to identify voiced regions
        frame_length = 2048
        hop_length = 512
        
        energy = []
        for i in range(0, len(self.signal) - frame_length, hop_length):
            frame = self.signal[i:i+frame_length]
            energy.append(np.sum(frame**2))
        
        energy = np.array(energy)
        energy = energy / (np.max(energy) + 1e-10)  # Normalize
        
        # Voicing threshold: identify voiced regions
        threshold = np.percentile(energy, threshold_percentile)
        voiced_mask = energy > threshold
        
        # Initialize pitch marks list
        pitch_mark_times = []
        pitch_mark_freqs = []
        
        # Interpolate F0 to full signal resolution for better precision
        from scipy.interpolate import interp1d
        f0_interp = interp1d(times, frequencies, kind='linear', fill_value='extrapolate')
        
        # Find first voiced region to start
        first_voiced_frame = np.where(voiced_mask)[0]
        if len(first_voiced_frame) == 0:
            print("No voiced regions detected")
            return np.array([]), np.array([])
        
        current_time = (first_voiced_frame[0] * hop_length + frame_length // 2) / self.fs
        
        # Generate pitch marks at 1/F0 intervals, aligned to peaks
        while current_time < self.duration:
            # Get F0 at current time
            f0_current = f0_interp(current_time)
            pitch_period = 1.0 / f0_current
            
            # Check if in voiced region
            frame_idx = int(current_time * self.fs / hop_length)
            if frame_idx < len(voiced_mask) and voiced_mask[frame_idx]:
                # Find maximum amplitude in this pitch period window
                window_start = current_time - pitch_period * 0.4
                window_end = current_time + pitch_period * 0.4
                
                start_sample = max(0, int(window_start * self.fs))
                end_sample = min(len(self.signal), int(window_end * self.fs))
                
                if start_sample < end_sample:
                    # Find peak (max absolute value)
                    signal_window = self.signal[start_sample:end_sample]
                    peak_idx = np.argmax(np.abs(signal_window))
                    peak_time = start_sample / self.fs + peak_idx / self.fs
                    
                    pitch_mark_times.append(peak_time)
                    pitch_mark_freqs.append(f0_current)
            
            # Advance by one pitch period
            current_time += pitch_period
        
        pitch_mark_times = np.array(pitch_mark_times)
        pitch_mark_freqs = np.array(pitch_mark_freqs)
        
        print(f"✓ Detected pitch marks: {len(pitch_mark_times)} marks (aligned to peaks)")
        if len(pitch_mark_times) > 0:
            avg_spacing_ms = (np.mean(np.diff(pitch_mark_times)) * 1000) if len(pitch_mark_times) > 1 else 0
            print(f"  Average pitch period: {avg_spacing_ms:.2f} ms")
            print(f"  First 10 marks at times: {pitch_mark_times[:10]}... (seconds)")
        
        return pitch_mark_times, pitch_mark_freqs
    
    def plot_analysis(self, times, frequencies, pitch_mark_times=None, pitch_mark_freqs=None):
        """
        Plot the fundamental frequency contour and pitch marks.
        
        Parameters:
        - times: Time array from fundamental frequency detection
        - frequencies: Fundamental frequency array
        - pitch_mark_times: Times of detected pitch marks
        - pitch_mark_freqs: F0 values at pitch marks
        """
        plt.figure(figsize=(14, 8))
        
        # Plot fundamental frequency
        plt.subplot(2, 1, 1)
        plt.plot(times, frequencies, 'b-', linewidth=1.5, label='Fundamental Frequency')
        if pitch_mark_times is not None and len(pitch_mark_times) > 0:
            plt.scatter(pitch_mark_times, pitch_mark_freqs, color='red', s=50, 
                       marker='v', label=f'Pitch Marks ({len(pitch_mark_times)})', zorder=5, alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Fundamental Frequency - {INSTRUMENT}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot spectrogram with overlay
        plt.subplot(2, 1, 2)
        nSegment = 512
        noverlap = int(nSegment * 0.75)
        stride = nSegment - noverlap
        
        segment_starts = range(0, len(self.signal) - nSegment + 1, stride)
        segments = np.array([self.signal[i:i + nSegment] for i in segment_starts])
        window_func = signal.get_window('hann', nSegment)
        ffts = np.fft.fft(segments * window_func, axis=1)
        magnitude = np.abs(ffts)
        magnitude = magnitude[:, :nSegment // 2]
        
        freqs = np.fft.fftfreq(nSegment, 1/self.fs)[:nSegment // 2]
        seg_times = np.array([i / self.fs for i in segment_starts])
        
        plt.pcolormesh(seg_times, freqs, 20 * np.log10(magnitude.T + 1e-10), 
                       shading='auto', cmap='viridis')
        plt.plot(times, frequencies, 'r-', linewidth=2, label='F0')
        if pitch_mark_times is not None and len(pitch_mark_times) > 0:
            plt.scatter(pitch_mark_times, pitch_mark_freqs, color='yellow', s=50, 
                       marker='v', label='Pitch Marks', zorder=5, alpha=0.8)
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram with Fundamental Frequency Overlay and Pitch Marks')
        plt.ylim([0, 1500])
        plt.legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = Path(self.wav_path).parent.parent / f'{INSTRUMENT}_analysis.png'
            plt.savefig(str(output_file), dpi=100, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_zoomed_time_view(self, pitch_mark_times, num_marks=5):
        """
        Plot a zoomed time-domain view showing pitch marks in detail.
        
        Parameters:
        - pitch_mark_times: Times of detected pitch marks
        - num_marks: Number of pitch marks to display (default 5)
        """
        if pitch_mark_times is None or len(pitch_mark_times) < num_marks:
            print("Not enough pitch marks to display zoomed view")
            return
        
        # Select a middle region with good signal (avoid start/end)
        start_idx = len(pitch_mark_times) // 2
        end_idx = start_idx + num_marks
        
        # Get time window around selected pitch marks
        center_time = pitch_mark_times[start_idx]
        mark_times = pitch_mark_times[start_idx:end_idx]
        
        # Calculate window around marks (1.5 pitch periods before first, after last)
        pitch_period = mark_times[1] - mark_times[0] if len(mark_times) > 1 else 0.004
        window_start = mark_times[0] - pitch_period * 1.5
        window_end = mark_times[-1] + pitch_period * 1.5
        
        # Extract signal in window
        start_sample = int(window_start * self.fs)
        end_sample = int(window_end * self.fs)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.signal), end_sample)
        
        signal_window = self.signal[start_sample:end_sample]
        time_window = np.arange(len(signal_window)) / self.fs + window_start
        
        # Create figure
        plt.figure(figsize=(14, 6))
        
        # Plot waveform
        plt.plot(time_window, signal_window, 'b-', linewidth=1.5, label='Audio Signal')
        
        # Plot pitch marks
        plt.scatter(mark_times, 
                   np.interp(mark_times, time_window, signal_window),
                   color='red', s=200, marker='v', label='Pitch Marks', 
                   zorder=5, edgecolors='darkred', linewidths=2)
        
        # Add vertical lines at pitch marks for clarity
        for i, t in enumerate(mark_times):
            plt.axvline(t, color='red', alpha=0.3, linestyle='--', linewidth=1)
            plt.text(t, plt.ylim()[1] * 0.95, f'M{i+1}', ha='center', 
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Zoomed Time-Domain View - {num_marks} Pitch Marks ({INSTRUMENT})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if self.save_plots:
            output_file = Path(self.wav_path).parent.parent / f'{INSTRUMENT}_zoomed_time.png'
            plt.savefig(str(output_file), dpi=100, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()


# ===================== MAIN EXECUTION =============================

if __name__ == "__main__":
    # Construct path to instrument wav file
    wav_file_path = Path(__file__).parent / 'InstrumentsPSOLA' / f'{INSTRUMENT}.wav'
    
    print(f"\n{'='*60}")
    print(f"PSOLA Pitch Detector - {INSTRUMENT}")
    print(f"{'='*60}\n")
    
    # Create detector (save_plots=True, show_plots=False for batch processing)
    detector = PSLAPitchDetector(str(wav_file_path), min_freq=50, max_freq=2000, 
                                  save_plots=True, show_plots=False)
    
    # Load audio
    if not detector.load_audio():
        exit(1)
    
    # Compute and display spectrogram
    print("\n[1/4] Computing spectrogram...")
    detector.compute_spectrogram(nSegment=512, overlap_percentage=75)
    
    # Detect fundamental frequency
    print("\n[2/4] Detecting fundamental frequency...")
    times, frequencies = detector.detect_fundamental_frequency(frame_length=4096, hop_length=512)
    
    # Detect pitch marks (one per pitch period)
    print("\n[3/4] Detecting pitch marks (1 per pitch period)...")
    pitch_mark_times, pitch_mark_freqs = detector.detect_pitch_marks(times, frequencies)
    
    # Plot results
    print("\n[4/4] Plotting analysis...")
    detector.plot_analysis(times, frequencies, pitch_mark_times, pitch_mark_freqs)
    
    # Plot zoomed time-domain view
    print("\n[5/5] Plotting zoomed time-domain view...")
    detector.plot_zoomed_time_view(pitch_mark_times, num_marks=5)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")
