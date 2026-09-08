from copyreg import pickle

import adi
import os
import time
import pickle
from datetime import datetime
from scipy import signal
import WaveformGenerator

import matplotlib.pyplot as plt
import numpy as np

from MatchFilter import matched_filter
from MatchFilter import range_matched_filter_1d

NUM_SAMPLES = int(1024*1024/4/4)
AMPLITUDE = 1.0
realtime = False  # Set to True for continuous operation, False for single run
cal = False  # Set to True to perform calibration and save data, False to use existing calibration data
import seaborn as sns
import scienceplots
plt.style.use(['ieee'])

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titlesize': 8,      # Reduced from 10
    'axes.labelsize': 10,      # Reduced from 10
    'xtick.labelsize': 10,     # Reduced from 8
    'ytick.labelsize': 10,     # Reduced from 8
    'legend.fontsize': 10,     # Reduced from 8
    'font.size': 6            # Reduced from 8 (affects colorbar)
})

dev = adi.DAQ2("ip:analog.local")
import numpy as np

def calculate_a_calibrated(a_IF, a_IF_c, t, f_IF_c=None, W=None, R_cal=None, c=299792458.0):
    """
    Calculates the calibrated signal a_calibrated(t) according to:
    a_calibrated(t) = [a_IF(t) / a_IF,c(t)] * exp(2 * pi * j * f_IF,c * t)
                    = [a_IF(t) / a_IF,c(t)] * exp((4 * pi * j * W * R_cal * t) / c)

    Parameters
    ----------
    a_IF : array_like
        Intermediate frequency signal a_IF(t).
    a_IF_c : array_like
        Calibration signal a_IF,c(t).
    t : array_like
        Time vector t (seconds).
    f_IF_c : float, optional
        Calibration intermediate frequency f_IF,c (Hz).
    W : float, optional
        Chirp sweep rate / frequency slope (Hz/s).
    R_cal : float, optional
        Distance to calibration target (meters).
    c : float, optional
        Speed of light in m/s (default is 299,792,458 m/s).

    Returns
    -------
    ndarray (complex)
        The calibrated signal a_calibrated(t).
    """
    # a_IF = np.asarray(a_IF, dtype=np.complex128)
    # a_IF_c = np.asarray(a_IF_c, dtype=np.complex128)
    # t = np.asarray(t)

    # Compute phase factor using either f_IF_c directly or via W and R_cal
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

run_plot = False

dev._ctx.set_timeout(3000)
dev._rxadc.set_kernel_buffers_count(1)
dev._txdac.set_kernel_buffers_count(2)
dev.rx_enabled_channels = [0, 1]
dev.tx_enabled_channels = [0, 1]
dev.rx_buffer_size = NUM_SAMPLES
dev.tx_cyclic_buffer = False
fs = int(dev.sample_rate)
dev.dds_single_tone(fs / 10, 0.0, channel=0)
print("Device Started")

generator = WaveformGenerator.WaveformGenerator(num_samples=int(NUM_SAMPLES/2), fs=fs, amplitude= 2 ** 13)

f_start_cycles_per_sample=-0.5  # Normalized start frequency (100 MHz / fs)
sweep_duration_samples=int(NUM_SAMPLES/2)  # Sweep duration in samples (50 us * fs)
bandwidth_cycles_per_sample=1  # Normalized bandwidth (5 MHz / fs)

FMCW = generator.generate_fmcw(f_start_cycles_per_sample=f_start_cycles_per_sample,  
                               sweep_duration_samples=sweep_duration_samples,  
                               bandwidth_cycles_per_sample=bandwidth_cycles_per_sample 
                               )[1]

real=np.real(FMCW)
imag=np.imag(FMCW)

tx_iq = [real, imag]
tx_iq_ref = tx_iq[1] + 1j * tx_iq[0]

# --- SPECTROGRAM for RX ---
WIN_LEN = 1024
win = signal.windows.hann(WIN_LEN)

HOP = (16)
SFT = signal.ShortTimeFFT(
    win,
    hop=HOP,
    fs=fs,
    fft_mode="centered",
    scale_to="psd",
)
print("Generated Waveform")
def set_window_position(fig, x, y):
    """Position a figure window at screen coordinates (x, y)."""
    try:
        # Works for Tkinter backend (default on Windows/Linux)
        fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    except AttributeError:
        try:
            # Works for PyQt / PySide backends
            fig.canvas.manager.window.move(x, y)
        except AttributeError:
            pass
if realtime:
    plt.ion()
    plt.rcParams['figure.dpi'] = 150  # Lower DPI = smaller display window on screen
    fig1, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    fig2, ax2 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    fig3, ax3 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    fig4, ax4 = plt.subplots(figsize=(4, 3), constrained_layout=True)

    # Recalculate positions based on figure size and DPI
    dpi = plt.rcParams['figure.dpi']
    fig_width_px = int(fig1.get_size_inches()[0] * dpi)
    fig_height_px = int(fig1.get_size_inches()[1] * dpi)
    gap_px = 50

    set_window_position(fig1, x=gap_px, y=gap_px)  # Top-Left
    set_window_position(fig2, x=gap_px + fig_width_px + gap_px, y=gap_px)  # Top-Right
    set_window_position(fig3, x=gap_px, y=gap_px + fig_height_px + gap_px)  # Bottom-Left
    set_window_position(fig4, x=gap_px + fig_width_px + gap_px, y=gap_px + fig_height_px + gap_px)  # Bottom-Right
run_index = 0
plot_fig = None

try:
    while True:
        start = time.perf_counter()
        dev.rx_sync_start = "arm"
        dev.tx_sync_start = "arm"

        if not ("arm" == dev.tx_sync_start == dev.rx_sync_start):
            raise Exception(
                "Unexpected SYNC status: TX "
                + dev.tx_sync_start
                + " RX: "
                + dev.rx_sync_start
            )
        if run_index == 0:
            dev.tx_destroy_buffer()
        dev.rx_destroy_buffer()
        dev._rx_init_channels()
        dev.tx(tx_iq)
        dev.tx_sync_start = "trigger_manual"

        if not ("disarm" == dev.tx_sync_start == dev.rx_sync_start):
            raise Exception(
                "Unexpected SYNC status: TX "
                + dev.tx_sync_start
                + " RX: "
                + dev.rx_sync_start
            )

        try:
            x = dev.rx()
        except Exception as e:
            print("Run#", run_index, " ----------------------------- FAILED:", e)
            run_index += 1
            continue
        end = time.perf_counter()

        print(f"Run {run_index+1} completed in {end - start:.4f} seconds")
        rx_real = x[1]
        rx_imag = x[0]
        rx_iq = (rx_real + 1j * rx_imag)*2**14

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        radar_data_path = os.path.join(data_dir, f'radar_data_{timestamp}.pkl')
        if not realtime:
            with open(radar_data_path, 'wb') as f:
                pickle.dump(rx_iq, f)

        radar_data_path = os.path.join(data_dir, "radar_data_20260731_214839.pkl")
        with open(radar_data_path, 'rb') as f:
             rx_iq = pickle.load(f)

        if_average = 0
        tx_iq_ref_padded = np.pad(tx_iq_ref[::-1], (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
        if_output  = rx_iq*tx_iq_ref_padded

        if cal:
            with open('radar_data.pkl', 'wb') as f:
                pickle.dump(if_output, f)
            break

        ########################
        with open('radar_data.pkl', 'rb') as f:
             a_IF_c = pickle.load(f)

        # Drop beginning and ending samples to remove transient artifacts
        trim_samples = min(100, len(if_output) // 10, len(a_IF_c) // 10)
        if trim_samples > 0:
            if_output = if_output[1000:40000]
            a_IF_c = a_IF_c[1000:40000]

        # Ensure both signals have the same length after trimming
        min_len = min(len(if_output), len(a_IF_c))
        if_output = if_output[:min_len]
        a_IF_c = a_IF_c[:min_len]

        num_samples = len(if_output)
        t = np.arange(num_samples) / fs

        bw = bandwidth_cycles_per_sample * fs
        print(f"Bandwidth (Hz): {bw}")
        duration = sweep_duration_samples / fs
        print(f"Sweep Duration (s): {duration}")
        W = bw / duration  # Chirp rate in Hz/s
        # if_output = calculate_a_calibrated(
        #     a_IF=if_output, 
        #     a_IF_c=a_IF_c, 
        #     t=t, 
        #     W=W,       # e.g., 1 THz/s chirp rate
        #     R_cal=3.0   
        # )


        ########################

        n_fft = len(if_output)
        n_fft_padded = n_fft *10**3 # Increase FFT size by factor of 4

        # Frequency and Range axis calculation
        #f_axis = np.fft.fftfreq(n_fft_padded, 1/fs)
        f_axis = -np.fft.fftshift(np.fft.fftfreq(n_fft_padded, 1/fs))

        range_axis = (f_axis * 299792458.0 ) / (2 * W)

        window = np.hanning(len(if_output))
        #fft_raw = np.fft.fft(if_output * window, n=n_fft_padded)
        fft_raw = np.fft.fftshift(np.fft.fft(if_output * window, n=n_fft_padded))
        fft_magnitude = np.abs(fft_raw)

        #if_output_mag_padded = np.pad(if_output_mag * window, (0, n_fft_padded - len(if_output_mag)), mode='constant')
        
        # Plot the IF output (magnitude) and I/Q traces in real time
        if run_plot:
            if plot_fig is None:
                plot_fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            else:
                for ax in axs.flat:
                    ax.clear()

            axs[0, 0].plot(real, label='Transmitted I')
            axs[0, 0].plot(imag, label='Transmitted Q')
            #axs[0, 0].set_title(f'Run {run_index+1} - Transmitted I/Q vs Sample')
            axs[0, 0].set_xlabel('Sample')
            axs[0, 0].set_ylabel('Amplitude')
            axs[0, 0].legend()
            axs[0, 0].grid(True)

            axs[0, 1].specgram(real, Fs=fs)
            #axs[0, 1].set_title(f'Run {run_index+1} - Transmitted I Spectrogram')
            axs[0, 1].set_ylabel('Frequency')

            axs[1, 0].plot(rx_real, label='Received I')
            axs[1, 0].plot(rx_imag, label='Received Q')
            #axs[1, 0].set_title(f'Run {run_index+1} - Received I/Q vs Sample')
            axs[1, 0].set_xlabel('Sample')
            axs[1, 0].set_ylabel('Amplitude')
            axs[1, 0].legend()
            axs[1, 0].grid(True)

            axs[1, 1].specgram(rx_real, Fs=fs)
            #axs[1, 1].set_title(f'Run {run_index+1} - Received I Spectrogram')
            axs[1, 1].set_ylabel('Frequency')

            plot_fig.canvas.draw()
            plot_fig.canvas.flush_events()
            plt.pause(0.01)
            if not plt.fignum_exists(plot_fig.number):
                print('Plot window closed, stopping realtime loop.')
                break

        range_profile_db = 20 * np.log10(fft_magnitude + 1e-12)
        range_profile_db -= np.max(range_profile_db)  # normalize the range profile so the highest peak is at 0 dB
        peak_threshold = np.max(range_profile_db) - 5
        peaks, properties = signal.find_peaks(
            range_profile_db,
            height=peak_threshold,
            distance=2
        )
        peak_ranges = range_axis[peaks]
        peak_values = properties["peak_heights"]

        print("Detected peaks (m, dB):", list(zip(np.round(peak_ranges, 2), np.round(peak_values, 1))))



        if realtime:
            # Clear existing content from persistent figures for live redraw
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
        else:
            # Create single-use figure instances for export mode
            fig1, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            fig2, ax2 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            fig3, ax3 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            fig4, ax4 = plt.subplots(figsize=(4, 3), constrained_layout=True)

        # --- 1. IF Output Plot ---
        ax1.plot(abs(if_output), label='IF output')
        #ax1.set_title(f'Run {run_index+1} - IF output')
        ax1.set_xlabel('Sample', fontweight='bold', labelpad=10)
        ax1.set_ylabel('Amplitude', fontweight='bold', labelpad=10)
        ax1.legend()
        ax1.grid(True)

        # --- 2. Range Profile Plot ---
        ax2.plot(range_axis, range_profile_db, label='Range profile (dB)')
        ax2.plot(peak_ranges, peak_values, "x", color="red", label="Peaks")
        for i, (x, y) in enumerate(zip(peak_ranges, peak_values)):
            ax2.annotate(
                f"{x:.1f} m",
                xy=(x, y),
                xytext=(-10, -(20 + 20 * i)),
                textcoords="offset points",
                fontweight='bold',
                ha="center",
                va="bottom",
                color="red",
                fontsize=20,
            )
        #ax2.set_title(f'Run {run_index+1} - Average Range Profile (matched filter via FFT)')
        ax2.set_xlabel('Range (m)', fontweight='bold', labelpad=10, fontsize=18)
        #ax2.set_xlim(1, 5)
        ax2.set_xlim(63, 68)
        ax2.set_ylim(-60, 1)
        ax2.set_ylabel('Amplitude (dB)', fontweight='bold', labelpad=10, fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.minorticks_on()
        #ax2.legend(loc="lower right",fontsize=18)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- 3. TX vs RX I Plot ---
        ax3.plot(real, label='TX I')
        ax3.plot(rx_real, label='RX I')
        #ax3.set_title(f'Run {run_index+1} - TX vs RX I')
        ax3.set_xlabel('Sample', fontweight='bold', labelpad=10, fontsize=20)
        ax3.set_ylabel('Amplitude', fontweight='bold', labelpad=10, fontsize=20)
        ax3.legend()
        ax3.grid(True)

        # --- 4. Spectrogram Plot ---
        ax4.specgram(rx_iq, Fs=fs)
        #ax4.set_title(f'Run {run_index+1} - Spectrogram')
        ax4.set_ylabel('Frequency', fontweight='bold', labelpad=10, fontsize=14)
        ax4.tick_params(axis='both', which='major', labelsize=12)

        # --- Realtime GUI Update vs Static Save Logic ---
        if realtime:
            # Pause briefly to yield execution and render live UI updates smoothly
            plt.pause(0.1)
        else:
            # Save plots to disk in non-realtime mode, then close and break
            fig1.savefig(f"radar_run_{run_index+1}_if_output.png", dpi=200, bbox_inches='tight')
            fig2.savefig(f"radar_run_{run_index+1}_range_profile.png", dpi=200, bbox_inches='tight')
            fig3.savefig(f"radar_run_{run_index+1}_tx_rx_i.png", dpi=200, bbox_inches='tight')
            fig4.savefig(f"radar_run_{run_index+1}_spectrogram.png", dpi=200, bbox_inches='tight')

            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            plt.close(fig4)
            break

        run_index += 1

except KeyboardInterrupt:
    print('Realtime plot stoppedl by user.')

# if run_plot and plot_fig is not None:
#     plt.ioff()
#     plt.show(block=True)

if False:
            # --- SPECTROGRAM for RX ---
            Sxx_rx = SFT.spectrogram(rx_iq)
            Sxx_dB_rx = 10 * np.log10(np.maximum(Sxx_rx, 1e-12))

            t = np.arange(Sxx_rx.shape[1]) * SFT.delta_t
            f = SFT.f
            
            # --- SPECTROGRAM for TX ---
            tx_iq_ref_padded = np.pad(tx_iq_ref[::-1], (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
            Sxx_tx = SFT.spectrogram(tx_iq_ref_padded)
            Sxx_dB_tx = 10 * np.log10(np.maximum(Sxx_tx, 1e-12))

            t = np.arange(Sxx_tx.shape[1]) * SFT.delta_t
            f = SFT.f

            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

            # RX
            im_rx = ax.pcolormesh(t, f / 1e6, Sxx_dB_rx, shading="auto", cmap="Greens")
            cbar = fig.colorbar(im_rx, ax=ax, label="Power [dB] (RX)")

            # TX overlaid (transparent)
            im_tx = ax.pcolormesh(
                t, f / 1e6, Sxx_dB_tx,
                shading="auto",
                cmap="Reds",
                alpha=0.45
            )
            # optional second colorbar (comment out if you don't want two)
            fig.colorbar(im_tx, ax=ax, label="Power [dB] (TX)")

            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [MHz]")
            ax.set_title("RX (viridis) overlaid with TX (magma) spectrogram")

            plt.show()

