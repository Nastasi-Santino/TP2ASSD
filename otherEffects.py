import numpy as np

def distorsion_effect(x, gain=2.0):
    y = gain * x
    y = np.clip(y, -1, 1)
    return y