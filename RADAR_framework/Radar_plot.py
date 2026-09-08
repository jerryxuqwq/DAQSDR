import os
import time
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import adi
import WaveformGenerator

# --- Plot Style Configuration ---
try:
    import scienceplots
    plt.style.use(['ieee'])
except ImportError:
    plt.style.use('default')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titlesize': 8,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.size': 6
})

# --- System Parameters ---
NUM_SAMPLES = int(1024 * 1024 / 4 / 4)
AMPLITUDE = 1.0
REALTIME = False    # Set to True for continuous operation, False for single run
CAL = False         # Set to True to perform calibration and save reference data
RUN_PLOT = False    # Toggle real-time streaming plot window
SPEED_OF_LIGHT = 299792458.0


def calculate_a_calibrated(a_IF, a_IF_c, t, f_IF_c=None, W=None, R_cal=None, c=SPEED_OF_LIGHT):
    """
    Calculates the calibrated intermediate frequency signal a_calibrated(t):
        a_calibrated(t) = [a_IF(t) / a_IF_c(t)] * exp(j * phi_cal)
    """
    if f_IF_c is not None:
        exponent = 2 * np.pi * 1j * f_IF_c * t
    elif W is not None and R_cal is not None:
        exponent = (4 * np.pi * 1j * W * R_cal * t) / c
    else:
        raise ValueError("You must provide either 'f_IF_c' or both 'W' and 'R_cal'.")

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(
            a_IF,
            a_IF_c,
            out=np.zeros_like(a_IF, dtype=np.complex128),
            where=a_IF_c != 0,
        )

    return ratio * np.exp(exponent)

def system_delay_remove(a_IF, shift, W, t=None, fs=None, c=SPEED_OF_LIGHT):
    """
    Removes system delay / distance shift (in meters) from an FMCW IF signal.

    Parameters
    ----------
    a_IF : np.ndarray
        Intermediate Frequency (IF) signal array.
    shift : float
        Shift distance to compensate in meters.
    W : float
        Chirp slope/rate K = Bandwidth / Chirp_Time (in Hz/s).
    t : np.ndarray, optional
        Time vector in seconds. If None, `fs` must be provided.
    fs : float, optional
        Sampling frequency in Hz (used to calculate `t` if `t` is None).
    c : float, optional
        Speed of light in m/s (default: 299792458.0).

    Returns
    -------
    a_calibrated : np.ndarray
        The calibrated/shifted IF signal.
    """
    # Calculate time vector if not explicitly provided
    if t is None:
        if fs is None:
            raise ValueError("You must provide either the time vector `t` or sampling frequency `fs`.")
        num_samples = a_IF.shape[-1]
        t = np.arange(num_samples) / fs

    # Beat frequency phase ramp for range offset: -j * 2 * pi * f_beat * t
    # where f_beat = (2 * W * shift) / c
    exponent = -1j * (4 * np.pi * W * shift * t) / c

    return a_IF * np.exp(exponent)


def compute_range_profile(if_signal, fs, W, n_fft_factor=100, c=SPEED_OF_LIGHT):
    """Computes range profile (dB) and range axis from an IF signal."""
    num_samples = len(if_signal)
    n_fft_padded = num_samples * n_fft_factor
    window = np.hanning(num_samples)

    f_axis = np.fft.fftshift(np.fft.fftfreq(n_fft_padded, 1 / fs))
    range_axis = (f_axis * c) / (2 * W)

    fft_raw = np.fft.fftshift(np.fft.fft(if_signal * window, n=n_fft_padded))
    fft_magnitude = np.abs(fft_raw)

    range_profile_db = 20 * np.log10(fft_magnitude + 1e-12)
    range_profile_db -= np.max(range_profile_db)  # Normalize peak to 0 dB

    return range_axis, range_profile_db


def set_window_position(fig, x, y):
    """Position a figure window at screen coordinates (x, y)."""
    try:
        fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    except AttributeError:
        try:
            fig.canvas.manager.window.move(x, y)
        except AttributeError:
            pass


# --- Device Setup ---
dev = adi.DAQ2("ip:analog.local")
dev._ctx.set_timeout(3000)
dev._rxadc.set_kernel_buffers_count(1)
dev._txdac.set_kernel_buffers_count(2)
dev.rx_enabled_channels = [0, 1]
dev.tx_enabled_channels = [0, 1]
dev.rx_buffer_size = NUM_SAMPLES
dev.tx_cyclic_buffer = False
fs = int(dev.sample_rate)
dev.dds_single_tone(fs / 10, 0.0, channel=0)
print("Device Initialized.")

# --- Waveform Generation ---
generator = WaveformGenerator.WaveformGenerator(num_samples=int(NUM_SAMPLES / 2), fs=fs, amplitude=2**13)

f_start_cycles_per_sample = -0.5
sweep_duration_samples = int(NUM_SAMPLES / 2)
bandwidth_cycles_per_sample = 1.0

FMCW = generator.generate_fmcw(
    f_start_cycles_per_sample=f_start_cycles_per_sample,
    sweep_duration_samples=sweep_duration_samples,
    bandwidth_cycles_per_sample=bandwidth_cycles_per_sample
)[1]

real = np.real(FMCW)
imag = np.imag(FMCW)
tx_iq = [real, imag]
tx_iq_ref = tx_iq[1] + 1j * tx_iq[0]

# --- Derived Chirp Parameters ---
bw = bandwidth_cycles_per_sample * fs
duration = sweep_duration_samples / fs
W = bw / duration  # Chirp rate in Hz/s
print(f"Bandwidth: {bw / 1e6:.2f} MHz | Duration: {duration * 1e6:.2f} us | Chirp Rate: {W:.2e} Hz/s")

# --- Directory Setup ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)
cal_file_path = os.path.join(data_dir, 'radar_cal_reference.pkl')

if REALTIME:
    plt.ion()
    plt.rcParams['figure.dpi'] = 150
    fig1, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    fig2, ax2 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    fig3, ax3 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    fig4, ax4 = plt.subplots(figsize=(4, 3), constrained_layout=True)

    dpi = plt.rcParams['figure.dpi']
    fig_w = int(fig1.get_size_inches()[0] * dpi)
    fig_h = int(fig1.get_size_inches()[1] * dpi)
    gap = 50

    set_window_position(fig1, gap, gap)
    set_window_position(fig2, gap * 2 + fig_w, gap)
    set_window_position(fig3, gap, gap * 2 + fig_h)
    set_window_position(fig4, gap * 2 + fig_w, gap * 2 + fig_h)

run_index = 0
plot_fig = None

try:
    while True:
        rx_iq = None
        radar_data_path = os.path.join(data_dir, "radar_data_20260731_214839.pkl")
        with open(radar_data_path, 'rb') as f:
            rx_iq = pickle.load(f)
        rx_real = np.real(rx_iq)
        rx_imag = np.imag(rx_iq)


        # Mixing (Dechirping) -> Uncalibrated IF Output
        tx_iq_ref_padded = np.pad(tx_iq_ref[::-1], (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
        if_uncal = rx_iq * tx_iq_ref_padded

        # Calibration Save Mode
        if CAL:
            with open(cal_file_path, 'wb') as f:
                pickle.dump(if_uncal, f)
            print(f"Calibration reference saved to {cal_file_path}")
            break

        # Trimming transients
        start_idx = min(1000, len(if_uncal) // 10)
        end_idx = min(40000, len(if_uncal))
        if_uncal = if_uncal[start_idx:end_idx]

        # Calibration Application Logic
        if_cal = None
        if os.path.exists(cal_file_path):
            with open(cal_file_path, 'rb') as f:
                a_IF_c = pickle.load(f)

            a_IF_c = a_IF_c[start_idx:end_idx]
            min_len = min(len(if_uncal), len(a_IF_c))
            if_uncal = if_uncal[:min_len]
            a_IF_c = a_IF_c[:min_len]

            t = np.arange(min_len) / fs
            # Perform phase & amplitude calibration
            if_cal = calculate_a_calibrated(a_IF=if_uncal, a_IF_c=a_IF_c, t=t, W=W, R_cal=3.0)
        else:
            print("Warning: Calibration file not found. Plotting uncalibrated data only.")

        if_uncal = system_delay_remove(if_uncal, shift=-68, W=W, fs=fs)

        # --- Compute Range Profiles ---
        range_axis, profile_uncal_db = compute_range_profile(if_uncal, fs, W)
        if if_cal is not None:
            _, profile_cal_db = compute_range_profile(if_cal, fs, W)

        # Peak Detection (on calibrated data if available, else uncalibrated)
        target_profile = profile_cal_db if if_cal is not None else profile_uncal_db
        peaks, properties = signal.find_peaks(target_profile, height=-5, distance=2)
        peak_ranges = range_axis[peaks]
        peak_values = properties["peak_heights"]
        print("Detected peaks (Range m, dB):", list(zip(np.round(peak_ranges, 2), np.round(peak_values, 1))))

        # --- Plotting ---
        if not REALTIME:
            fig1, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            fig2, ax2 = plt.subplots(figsize=(8, 3), constrained_layout=True)
            fig3, ax3 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            fig4, ax4 = plt.subplots(figsize=(4, 3), constrained_layout=True)
        else:
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()

        # 1. IF Time-Domain Signal
        ax1.plot(np.abs(if_uncal), label='Uncalibrated IF')
        if if_cal is not None:
            ax1.plot(np.abs(if_cal), label='Calibrated IF', alpha=0.7)
        ax1.set_xlabel('Sample', fontweight='bold', labelpad=10)
        ax1.set_ylabel('Amplitude', fontweight='bold', labelpad=10)
        ax1.legend()
        ax1.grid(True)

        # 2. Combined Range Profile Plot (Uncalibrated vs Calibrated)
        ax2.plot(range_axis, profile_uncal_db, label='Uncalibrated', color='tab:blue', linestyle='--', linewidth=2)
        if if_cal is not None:
            ax2.plot(range_axis, profile_cal_db, label='Calibrated', color='tab:orange', linewidth=2)

        ax2.plot(peak_ranges, peak_values, "X", color="red", label="Peaks")
        for i, (rx_val, ry_val) in enumerate(zip(peak_ranges, peak_values)):
            ax2.annotate(
                f"{rx_val:.1f} m",
                xy=(rx_val, ry_val),                     # Peak point
                xytext=(0, -(120 + 20 * i)),             # Offset text location
                textcoords="offset points",
                fontweight='bold',
                ha="center",
                va="bottom",
                color="red",
                fontsize=20,
                arrowprops=dict(
                    arrowstyle="-",                      # Simple line (no arrow head)
                    color="red",
                    linestyle="--",                      # Dashed line
                    linewidth=1.5
                )
            )
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.minorticks_on()
        ax2.set_xlim(1, 5)
        ax2.set_ylim(-30, 1)
        ax2.legend(loc="lower right", fontsize=18)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Range (m)', fontweight='bold', labelpad=10, fontsize=18)
        ax2.set_ylabel('Amplitude (dB)', fontweight='bold', labelpad=10, fontsize=18)

        # 3. Transmitted vs Received I Plot
        ax3.plot(real[:1000], label='TX I')
        ax3.plot(rx_real[:1000], label='RX I', alpha=0.7)
        ax3.set_xlabel('Sample', fontweight='bold', labelpad=10)
        ax3.set_ylabel('Amplitude', fontweight='bold', labelpad=10)
        ax3.legend()
        ax3.grid(True)

        # 4. Spectrogram
        ax4.specgram(rx_iq, Fs=fs)
        ax4.set_ylabel('Frequency (Hz)', fontweight='bold', labelpad=10)

        # Output Handling
        if REALTIME:
            plt.pause(0.1)
        else:
            fig1.savefig(f"radar_run_{run_index+1}_if_output.png", dpi=200, bbox_inches='tight')
            fig2.tight_layout()
            fig2.savefig(f"radar_run_{run_index+1}_range_profile_comparison.png", dpi=200, bbox_inches='tight')
            fig3.savefig(f"radar_run_{run_index+1}_tx_rx_i.png", dpi=200, bbox_inches='tight')
            fig4.savefig(f"radar_run_{run_index+1}_spectrogram.png", dpi=200, bbox_inches='tight')

            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            plt.close(fig4)
            break

        run_index += 1

except KeyboardInterrupt:
    print('Realtime loop terminated by user.')