import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- PARAMETERS ---
filename = 'test.dat'
sample_rate = 500_000_000  # <-- Set your sample rate (Hz) here!

# --- LOAD DATA ---
data = np.fromfile(filename, dtype=np.int16)
if len(data) % 2 != 0:
    raise ValueError("File does not contain pairs of shorts.")
data = data.reshape(-1, 2)

channel1 = data[:, 0]  # I signal
channel2 = data[:, 1]  # Q signal

with open("received_path.pki", 'rb') as pkl_file:
    loaded_dict = pickle.load(pkl_file)
print(loaded_dict)

# --- SPECTROGRAM ---
WIN_LEN = 512
win = signal.windows.hann(WIN_LEN)

HOP = (8192)#125#256
SFT = signal.ShortTimeFFT(
    win,
    hop=HOP,
    fs=sample_rate,
    fft_mode="centered",
    scale_to="psd",
)

Sxx = SFT.spectrogram(channel1+1j*channel2)
Sxx_dB = 10 * np.log10(np.maximum(Sxx, 1e-12))

t = np.arange(Sxx.shape[1]) * SFT.delta_t
f = SFT.f

plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f / 1e6, Sxx_dB, shading="auto")
plt.colorbar(label="Power [dB]")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [MHz]")
plt.tight_layout()
plt.show()