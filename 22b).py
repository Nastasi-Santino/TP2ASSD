import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================
# Parámetros modificables
# =========================
L = 10
R_L = 0.8

# H(z) = 0.5(1 + z^-1) / [1 - 0.5 R_L z^-L (1 + z^-1)]
#
# Numerador: 0.5 + 0.5 z^-1
b = np.array([0.5, 0.5])

# Denominador:
# 1 - 0.5 R_L z^-L - 0.5 R_L z^-(L+1)
a = np.zeros(L + 2)
a[0] = 1
a[L] = -0.5 * R_L
a[L + 1] = -0.5 * R_L

# =========================
# Polos y ceros
# =========================
zeros = np.roots(b)
poles = np.roots(a)

plt.figure(figsize=(6, 6))

theta = np.linspace(0, 2*np.pi, 1000)
plt.plot(np.cos(theta), np.sin(theta), '--', label='Círculo unitario')

plt.scatter(np.real(zeros), np.imag(zeros), marker='o', s=80, label='Ceros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', s=80, label='Polos')

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel('Parte real')
plt.ylabel('Parte imaginaria')
plt.title(f'Polos y ceros para L = {L}, R_L = {R_L}')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# =========================
# Respuesta en frecuencia
# =========================
w, h = signal.freqz(b, a, worN=4096)

plt.figure(figsize=(9, 5))
plt.plot(w / np.pi, 20*np.log10(np.abs(h) + 1e-12))
plt.xlabel(r'Frecuencia normalizada $\omega/\pi$')
plt.ylabel('Magnitud [dB]')
plt.title(f'Respuesta en frecuencia |H(e^{{jω}})| para L = {L}, R_L = {R_L}')
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(w / np.pi, np.unwrap(np.angle(h)))
plt.xlabel(r'Frecuencia normalizada $\omega/\pi$')
plt.ylabel('Fase [rad]')
plt.title('Fase de la respuesta en frecuencia')
plt.grid(True)
plt.show()