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

try:
    import soundfile as sf
    USE_SOUNDFILE = True
except ImportError:
    USE_SOUNDFILE = False

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
    
    def detect_sustain_region(self, times, frequencies, f0_stability_window=0.5):
        """
        Detect the sustain region by finding where F0 changes are minimal.
        Skips attack (rising pitch) and release (falling pitch).
        
        Parameters:
        - times: Time array from fundamental frequency detection
        - frequencies: Fundamental frequency array
        - f0_stability_window: Window size (s) for smoothness calculation
        
        Returns:
        - sustain_start: Start time of sustain region (seconds)
        - sustain_end: End time of sustain region (seconds)
        """
        if len(times) < 5:
            return times[0], times[-1]
        
        # Calculate F0 change rate (derivative) in sliding windows
        window_samples = max(2, int(f0_stability_window / (times[1] - times[0]) if len(times) > 1 else 10))
        
        f0_change = []
        change_times = []
        
        for i in range(len(frequencies) - window_samples):
            window_f0 = frequencies[i:i+window_samples]
            # Use maximum change within window as metric
            change = np.max(np.abs(np.diff(window_f0)))
            f0_change.append(change)
            change_times.append(times[i + window_samples//2])
        
        f0_change = np.array(f0_change)
        change_times = np.array(change_times)
        
        # Smooth the change rate
        if len(f0_change) > 5:
            f0_change = signal.savgol_filter(f0_change, window_length=min(11, len(f0_change)//2*2+1), polyorder=2)
        
        # Find regions with MINIMAL F0 change (stable/sustain regions)
        change_threshold = np.percentile(f0_change, 35)  # Bottom 35% - smoother regions (balanced threshold)
        smooth_regions = f0_change < change_threshold
        
        # Find the longest continuous smooth region
        if not np.any(smooth_regions):
            # Fallback: use middle 60% of signal
            start_idx = int(len(frequencies) * 0.2)
            end_idx = int(len(frequencies) * 0.8)
            print(f"  No smooth region found, using middle 60% of signal")
            return times[start_idx], times[end_idx]
        
        # Find consecutive True regions
        diff = np.diff(smooth_regions.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if len(starts) == 0:
            starts = [0]
        if len(ends) == 0:
            ends = [len(smooth_regions)]
        
        # Find longest smooth region with MINIMUM LENGTH requirement
        max_length = 0
        best_start = 0
        best_end = 0
        min_time_length = 0.5  # Minimum 0.5 seconds for sustain region
        min_samples = max(5, int(min_time_length / (change_times[1] - change_times[0]) if len(change_times) > 1 else 10))
        
        for start, end in zip(starts, ends):
            length = end - start
            if length > max_length and length >= min_samples:  # Enforce minimum length
                max_length = length
                best_start = start
                best_end = end
        
        # If no region meets minimum, use middle portion
        if max_length < min_samples:
            print(f"  No sustain region >= {min_time_length}s found, using middle 50% of signal")
            start_idx = int(len(frequencies) * 0.25)
            end_idx = int(len(frequencies) * 0.75)
            return times[start_idx], times[end_idx]
        
        sustain_start = change_times[best_start]
        sustain_end = change_times[min(best_end, len(change_times) - 1)]
        
        print(f"  Sustain region: {sustain_start:.2f}s - {sustain_end:.2f}s ({sustain_end - sustain_start:.2f}s)")
        
        return sustain_start, sustain_end
    
    def detect_pitch_marks(self, times, frequencies, threshold_percentile=40, min_distance_ms=20, 
                           use_sustain_only=True):
        """
        Detect pitch marks (one per pitch period for PSOLA).
        Places marks at maximum amplitude within each pitch period.
        
        Parameters:
        - times: Time array from fundamental frequency detection
        - frequencies: Fundamental frequency array
        - threshold_percentile: Percentile threshold for voiced/unvoiced decision (lower = more lenient)
        - min_distance_ms: Minimum distance between pitch marks (unused, kept for compatibility)
        - use_sustain_only: If True, only detect marks in sustain region (skip attack/release)
        
        Returns:
        - pitch_marks: Time positions of all detected pitch marks
        - pitch_mark_frequencies: F0 at each pitch mark
        """
        if times is None or frequencies is None:
            print("Fundamental frequency not detected. Call detect_fundamental_frequency() first.")
            return None, None
        
        # Detect sustain region if requested
        sustain_start, sustain_end = 0.0, self.duration
        if use_sustain_only:
            print("  Detecting sustain region...")
            sustain_start, sustain_end = self.detect_sustain_region(times, frequencies)
        
        # Compute energy in each frame to identify voiced regions
        frame_length = 2048
        hop_length = 512
        
        energy = []
        for i in range(0, len(self.signal) - frame_length, hop_length):
            frame = self.signal[i:i+frame_length]
            energy.append(np.sum(frame**2))
        
        energy = np.array(energy)
        energy = energy / (np.max(energy) + 1e-10)  # Normalize
        
        # Voicing threshold: identify voiced regions (more lenient now)
        threshold = np.percentile(energy, threshold_percentile)
        voiced_mask = energy > threshold
        
        # Initialize pitch marks list
        pitch_mark_times = []
        pitch_mark_freqs = []
        
        # Interpolate F0 to full signal resolution for better precision
        from scipy.interpolate import interp1d
        f0_interp = interp1d(times, frequencies, kind='linear', fill_value='extrapolate')
        
        # Get mean F0 for reasonable range checking
        mean_f0 = np.mean(frequencies)
        f0_min = mean_f0 * 0.85  # Allow ±15% deviation
        f0_max = mean_f0 * 1.15
        
        # Start from beginning of sustain region
        current_time = sustain_start
        
        # Generate pitch marks at 1/F0 intervals, aligned to peaks
        while current_time < sustain_end:
            # Get F0 at current time
            f0_current = f0_interp(current_time)
            
            # Check if F0 is reasonable (valid and in expected range)
            if not (f0_min <= f0_current <= f0_max):
                # Move forward quickly if F0 is unreasonable
                current_time += 0.01  # 10ms step
                continue
            
            pitch_period = 1.0 / f0_current
            
            # Check if in voiced region (more lenient)
            frame_idx = int(current_time * self.fs / hop_length)
            is_voiced = (frame_idx < len(voiced_mask) and voiced_mask[frame_idx])
            
            # Also consider it voiced if F0 is stable (include low-energy pitched regions)
            if not is_voiced and f0_min <= f0_current <= f0_max:
                is_voiced = True
            
            if is_voiced:
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
        
        region_info = f"sustain region ({sustain_start:.2f}-{sustain_end:.2f}s)" if use_sustain_only else "full signal"
        print(f"✓ Detected pitch marks: {len(pitch_mark_times)} marks in {region_info} (F0 range: {f0_min:.1f}-{f0_max:.1f} Hz)")
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
    
    def save_cropped_audio_and_marks(self, sustain_start, sustain_end, pitch_mark_times):
        """
        Save cropped audio (sustain region only) and pitch mark indices.
        
        Parameters:
        - sustain_start: Start time of sustain region (seconds)
        - sustain_end: End time of sustain region (seconds)
        - pitch_mark_times: Times of detected pitch marks (seconds)
        """
        # Create output folder
        output_dir = Path(self.wav_path).parent.parent / 'PitchMarksPSOLA'
        output_dir.mkdir(exist_ok=True)
        
        # Extract cropped audio
        start_sample = int(sustain_start * self.fs)
        end_sample = int(sustain_end * self.fs)
        cropped_signal = self.signal[start_sample:end_sample]
        
        # Save cropped audio as WAV
        wav_output_file = output_dir / f'{INSTRUMENT}_cropped.wav'
        
        if USE_SOUNDFILE:
            sf.write(str(wav_output_file), cropped_signal, self.fs)
        else:
            # Fallback: use scipy
            # Convert to int16 for WAV
            cropped_int16 = np.int16(cropped_signal * 32767)
            from scipy.io import wavfile as wf
            wf.write(str(wav_output_file), self.fs, cropped_int16)
        
        print(f"✓ Saved cropped audio: {wav_output_file}")
        print(f"  Duration: {len(cropped_signal) / self.fs:.2f}s ({len(cropped_signal)} samples)")
        
        # Calculate pitch mark indices in cropped audio
        pitch_mark_indices = []
        for mark_time in pitch_mark_times:
            # Convert time to sample index in cropped audio
            sample_idx = int((mark_time - sustain_start) * self.fs)
            # Verify index is within bounds
            if 0 <= sample_idx < len(cropped_signal):
                pitch_mark_indices.append(sample_idx)
        
        pitch_mark_indices = np.array(pitch_mark_indices)
        
        # Save pitch mark indices as .npy (binary format, easy to load)
        npy_output_file = output_dir / f'{INSTRUMENT}_pitch_marks_indices.npy'
        np.save(str(npy_output_file), pitch_mark_indices)
        
        print(f"✓ Saved pitch mark indices (.npy): {npy_output_file}")
        print(f"  Pitch marks: {len(pitch_mark_indices)} marks")
        print(f"  Indices: {pitch_mark_indices[:20]}... (showing first 20)")
        
        # Also save as CSV for human readability
        csv_output_file = output_dir / f'{INSTRUMENT}_pitch_marks_indices.csv'
        np.savetxt(str(csv_output_file), pitch_mark_indices, fmt='%d', header='pitch_mark_index')
        
        print(f"✓ Saved pitch mark indices (.csv): {csv_output_file}")
        
        # Save metadata as text
        meta_output_file = output_dir / f'{INSTRUMENT}_metadata.txt'
        with open(meta_output_file, 'w') as f:
            f.write(f"PSOLA Pitch Detector - {INSTRUMENT}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Original Audio:\n")
            f.write(f"  Sampling frequency: {self.fs} Hz\n")
            f.write(f"  Total duration: {self.duration:.2f} seconds\n")
            f.write(f"  Total samples: {len(self.signal)}\n\n")
            f.write(f"Cropped Audio (Sustain Region):\n")
            f.write(f"  Start time: {sustain_start:.2f} seconds\n")
            f.write(f"  End time: {sustain_end:.2f} seconds\n")
            f.write(f"  Duration: {sustain_end - sustain_start:.2f} seconds\n")
            f.write(f"  Start sample: {start_sample}\n")
            f.write(f"  End sample: {end_sample}\n")
            f.write(f"  Total samples: {len(cropped_signal)}\n\n")
            f.write(f"Pitch Marks:\n")
            f.write(f"  Total pitch marks: {len(pitch_mark_indices)}\n")
            f.write(f"  Average spacing: {np.mean(np.diff(pitch_mark_indices)) if len(pitch_mark_indices) > 1 else 0:.1f} samples\n")
            f.write(f"  Min index: {np.min(pitch_mark_indices) if len(pitch_mark_indices) > 0 else 'N/A'}\n")
            f.write(f"  Max index: {np.max(pitch_mark_indices) if len(pitch_mark_indices) > 0 else 'N/A'}\n\n")
            f.write(f"How to use in Python:\n")
            f.write(f"  import numpy as np\n")
            f.write(f"  import soundfile as sf\n\n")
            f.write(f"  # Load cropped audio\n")
            f.write(f"  audio, sr = sf.read('{INSTRUMENT}_cropped.wav')\n\n")
            f.write(f"  # Load pitch mark indices\n")
            f.write(f"  marks = np.load('{INSTRUMENT}_pitch_marks_indices.npy')\n\n")
            f.write(f"  # Access audio at pitch marks\n")
            f.write(f"  for i, mark_idx in enumerate(marks):\n")
            f.write(f"      print(f'Pitch mark {{i}} at sample {{mark_idx}}: audio[{{mark_idx}}] = {{audio[mark_idx]}}')\n")
        
        print(f"✓ Saved metadata: {meta_output_file}")
        
        return cropped_signal, pitch_mark_indices
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
    
    # Detect pitch marks (one per pitch period, sustain only)
    print("\n[3/4] Detecting pitch marks (1 per pitch period, sustain region only)...")
    pitch_mark_times, pitch_mark_freqs = detector.detect_pitch_marks(times, frequencies, use_sustain_only=True)
    
    # Get sustain region for saving
    print("\n[4/4] Detecting sustain region for export...")
    sustain_start, sustain_end = detector.detect_sustain_region(times, frequencies)
    
    # Plot results
    print("\n[5/5] Plotting analysis...")
    detector.plot_analysis(times, frequencies, pitch_mark_times, pitch_mark_freqs)
    
    # Plot zoomed time-domain view
    print("\n[6/6] Plotting zoomed time-domain view...")
    detector.plot_zoomed_time_view(pitch_mark_times, num_marks=5)
    
    # Save cropped audio and pitch mark indices
    print("\n[7/7] Saving cropped audio and pitch mark indices...")
    cropped_audio, pitch_indices = detector.save_cropped_audio_and_marks(sustain_start, sustain_end, pitch_mark_times)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")
