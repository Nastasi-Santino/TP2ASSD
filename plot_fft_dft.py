import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('fft_dft_comparison.csv')

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
