import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from sfcw import generate_sfcw  # Assuming this is a custom function for generating SFCW signals
from MarkovChain import *
from scipy.signal.windows import gaussian

fs = 10e6            # 10 MHz sampling rate
# f_start = 1e6        # 1 MHz start frequency
# f_step = 25e3        # 25 kHz frequency step
# num_steps = 64       # 64 frequency tones
# step_duration = 40e-6  # 40 µs per tone

idle = State(
    "IDLE",
    f_start=0,
    f_step=0,
    num_steps=64,
    step_duration=40e-6
)

s1 = State(
    "state 1",
    f_start=2e6,
    f_step=30e3,
    num_steps=80,
    step_duration=50e-6
)

s2 = State(
    "state 2",
    f_start=3e6,
    f_step=35e3,
    num_steps=60,
    step_duration=100e-6
)

s3 = State(
    "state 3",
    f_start=4e6,
    f_step=40e3,
    num_steps=112,
    step_duration=70e-6
)

states = [idle, s1, s2, s3]

P = [
    [0.8, 0.2, 0.0, 0.0],
    [0.1, 0.7, 0.2, 0.0],
    [0.0, 0.3, 0.7, 0.0],
    [0.0, 0.0, 0.2, 0.8]
]

mc = MarkovChain(states, P)
total_sfcw = np.zeros(0, dtype=np.complex128)
total_t = np.zeros(0)
path = mc.simulate(start_state=idle, n_steps=50)
for s in path:
    print(s)
    f_start = s.get_var("f_start")
    f_step = s.get_var("f_step")
    num_steps = s.get_var("num_steps")
    step_duration = s.get_var("step_duration")

    t, sfcw, freqs = generate_sfcw(
    f_start=f_start,
    f_step=f_step,
    num_steps=num_steps,
    step_duration=step_duration,
    fs=fs,
    amplitude=1.0
    )

    total_sfcw = np.append(total_sfcw, sfcw)
    total_t = np.append(total_t, t)


# Add white Gaussian noise to the generated SFCW signal
snr_db = 80  # desired SNR in dB; lower -> noisier
rng = np.random.default_rng(123)  # change or remove seed for non-deterministic noise
sig_power = np.mean(np.abs(total_sfcw) ** 2)
noise_power = sig_power / (10 ** (snr_db / 10.0))
noise_std = np.sqrt(noise_power)
# Use real-valued noise since spectrogram uses the real part later
noise = noise_std * rng.standard_normal(size=total_sfcw.shape)
total_sfcw = total_sfcw + noise

# STFT parameters
win_len = 1024
hop = 512
nfft = 1024
w = gaussian(200, std=8, sym=True)  # symmetric Gaussian window
# Create STFT object
stft = ShortTimeFFT(
    win=w,
    hop=hop,
    fs=fs,
    fft_mode="centered"
    #mfft=nfft
)

# Compute STFT (complex)
Sxx = stft.stft(total_sfcw)   # shape: (freq, time)

# Time & frequency axes
t_spec = stft.t(len(total_sfcw))   # seconds
f = stft.f                          # Hz

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(
    20 * np.log10(np.abs(Sxx) + 1e-12),
    aspect='auto',
    origin='lower',
    #extent=[t_spec[0]*1e3, t_spec[-1]*1e3, f[0]*1e-6, f[-1]*1e-6],
    cmap='viridis'
)

plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.title("SFCW Spectrogram")
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.show()