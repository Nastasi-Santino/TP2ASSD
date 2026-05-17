import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('fft_dft_comparison.csv')

# Calculate magnitude for both FFT and DFT
df['FFT_Magnitude'] = np.sqrt(df['FFT_Real']**2 + df['FFT_Imag']**2)
df['DFT_Magnitude'] = np.sqrt(df['DFT_Real']**2 + df['DFT_Imag']**2)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot Real Part
ax1.plot(df['Index'], df['FFT_Real'], 'b-o', label='FFT Real', linewidth=2, markersize=4)
ax1.plot(df['Index'], df['DFT_Real'], 'r--s', label='DFT Real', linewidth=2, markersize=4)
ax1.set_xlabel('Frequency Bin')
ax1.set_ylabel('Real Part')
ax1.set_title('FFT vs DFT - Real Part')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Imaginary Part
ax2.plot(df['Index'], df['FFT_Imag'], 'b-o', label='FFT Imaginary', linewidth=2, markersize=4)
ax2.plot(df['Index'], df['DFT_Imag'], 'r--s', label='DFT Imaginary', linewidth=2, markersize=4)
ax2.set_xlabel('Frequency Bin')
ax2.set_ylabel('Imaginary Part')
ax2.set_title('FFT vs DFT - Imaginary Part')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fft_dft_comparison.png', dpi=150)
print("Graph saved as fft_dft_comparison.png")
plt.show()

# Create separate figure for Magnitude
fig2, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(df['Index'], df['FFT_Magnitude']/32768, 'b-o', label='FFT Magnitude', linewidth=2, markersize=4)
ax3.plot(df['Index'], df['DFT_Magnitude']/32768, 'r--s', label='DFT Magnitude', linewidth=2, markersize=4)
ax3.set_xlabel('Frequency Bin')
ax3.set_ylabel('Magnitude')
ax3.set_title('FFT vs DFT - Magnitude')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fft_dft_magnitude.png', dpi=150)
print("Magnitude graph saved as fft_dft_magnitude.png")
plt.show()
