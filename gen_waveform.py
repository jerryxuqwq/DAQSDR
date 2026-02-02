import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.signal import medfilt

from sfcw import generate_sfcw
from MarkovChain import *

# ============================================================
# PARAMETERS
# ============================================================

fs = 10e6          # Sampling rate (Hz)
snr_db = -10        # SNR (dB)
rng = np.random.default_rng(123)

# ============================================================
# MARKOV STATES (SFCW MODES)
# ============================================================

idle = State(
    "IDLE",
    f_start=0,
    f_step=0,
    num_steps=64,
    step_duration=40e-6
)

s1 = State(
    "STATE_1",
    f_start=2e6,
    f_step=30e3,
    num_steps=80,
    step_duration=50e-6
)

s2 = State(
    "STATE_2",
    f_start=3e6,
    f_step=35e3,
    num_steps=60,
    step_duration=100e-6
)

s3 = State(
    "STATE_3",
    f_start=4e6,
    f_step=40e3,
    num_steps=112,
    step_duration=70e-6
)

states = [idle, s1, s2, s3]

P = [
    [0.8, 0.2, 0.0, 0.0],
    [0.5, 0.3, 0.2, 0.0],
    [0.7, 0.1, 0.1, 0.1],
    [0.6, 0.0, 0.2, 0.2]
]

mc = MarkovChain(states, P)

# ============================================================
# GENERATE COMPOSITE SFCW SIGNAL
# ============================================================

total_sfcw = np.zeros(0, dtype=np.complex128)
total_t = np.zeros(0)

path = mc.simulate(start_state=idle, n_steps=50)

for s in path:
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

# ============================================================
# ADD AWGN
# ============================================================

sig_power = np.mean(np.abs(total_sfcw) ** 2)
noise_power = sig_power / (10 ** (snr_db / 10))
noise = np.sqrt(noise_power) * rng.standard_normal(total_sfcw.shape)
total_sfcw += noise

# ============================================================
# STFT / SPECTROGRAM
# ============================================================

win = gaussian(200, std=8, sym=True)
stft = ShortTimeFFT(
    win=win,
    hop=512,
    fs=fs,
    fft_mode="centered"
)

Sxx = stft.stft(total_sfcw)
Sxx_mag = np.abs(Sxx)

t_spec = stft.t(len(total_sfcw))
f = stft.f

# ============================================================
# RIDGE EXTRACTION (DOMINANT FREQUENCY VS TIME)
# ============================================================

ridge_idx = np.argmax(Sxx_mag, axis=0)
ridge_freq = f[ridge_idx]          # Hz
ridge_freq_s = medfilt(ridge_freq, kernel_size=9)

# ============================================================
# SEGMENTATION BASED ON FREQUENCY JUMPS
# ============================================================

df = np.abs(np.diff(ridge_freq_s))

# Robust adaptive threshold
df_thresh = 5 * np.median(df)

change_points = np.where(df > df_thresh)[0]

segments = np.split(np.arange(len(ridge_freq_s)), change_points + 1)

# ============================================================
# PARAMETER ESTIMATION PER SEGMENT
# ============================================================

results = []

for seg in segments:
    if len(seg) < 10:
        continue

    t_start = t_spec[seg[0]]
    t_end = t_spec[seg[-1]]
    duration = t_end - t_start

    freqs_seg = ridge_freq_s[seg]
    bandwidth = freqs_seg.max() - freqs_seg.min()

    results.append({
        "start_ms": t_start * 1e3,
        "duration_ms": duration * 1e3,
        "bandwidth_MHz": bandwidth / 1e6
    })

# ============================================================
# PRINT RESULTS
# ============================================================

print("\nDetected SFCW segments:\n")
print(results
)
for i, r in enumerate(results, 1):
    print(
        f"Segment {i:02d} | "
        f"Start: {r['start_ms']:8.2f} ms | "
        f"Duration: {r['duration_ms']:7.2f} ms | "
        f"Bandwidth: {r['bandwidth_MHz']:7.2f} MHz"
    )

# ============================================================
# PLOTS
# ============================================================

# --- Spectrogram ---
plt.figure(figsize=(11, 5))
plt.imshow(
    20 * np.log10(Sxx_mag + 1e-12),
    aspect="auto",
    origin="lower",
    extent=[t_spec[0]*1e3, t_spec[-1]*1e3, f[0]*1e-6, f[-1]*1e-6],
    cmap="viridis"
)
plt.plot(t_spec * 1e3, ridge_freq_s / 1e6, "r", linewidth=2)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.title("SFCW Spectrogram with Ridge Tracking")
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.show()

# --- Ridge + segmentation ---
plt.figure(figsize=(11, 4))
plt.plot(t_spec * 1e3, ridge_freq_s / 1e6, linewidth=2)

for r in results:
    plt.axvline(r["start_ms"], color="r", linestyle="--")

plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.title("Ridge-Based SFCW Segmentation")
plt.grid()
plt.tight_layout()
plt.show()
