import adi
import time

from scipy import signal
import WaveformGenerator

import matplotlib.pyplot as plt
import numpy as np

from MatchFilter import matched_filter
from MatchFilter import range_matched_filter_1d

NUM_SAMPLES = int(1024*1024/4/4)
AMPLITUDE = 1.0

dev = adi.DAQ2("ip:analog.local")


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

# pulse = generator.generate_pulse(fc_normalized=fs/fs,  # Normalized carrier frequency (fs / fs = 1.0)
#                               pulse_width_samples=1024,
#                               pri_samples=2048
#                               )[1] 
f_start_cycles_per_sample=0.1  # Normalized start frequency (100 MHz / fs)
sweep_duration_samples=int(NUM_SAMPLES/2)  # Sweep duration in samples (50 us * fs)
bandwidth_cycles_per_sample=0.4  # Normalized bandwidth (5 MHz / fs)

FMCW = generator.generate_fmcw(f_start_cycles_per_sample=f_start_cycles_per_sample,  
                               sweep_duration_samples=sweep_duration_samples,  
                               bandwidth_cycles_per_sample=bandwidth_cycles_per_sample 
                               )[1]

real=np.real(FMCW)
imag=np.imag(FMCW)

tx_iq = [real, imag]
tx_iq_ref = tx_iq[0] + 1j * tx_iq[1]

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

plt.ion()
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
        rx_real = x[0]
        rx_imag = x[1]
        rx_iq = rx_real + 1j * rx_imag
        if_average = 0
        tx_iq_ref_padded = np.pad(tx_iq_ref[::-1], (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
        if_output  = rx_iq*tx_iq_ref_padded
            # FFT Processing
        if_output_mag=np.abs(if_output) 
        n_fft = len(if_output_mag)
        # Zero padding to increase FFT bin resolution
        n_fft_padded = n_fft * 4  # Increase FFT size by factor of 4
        f_axis = np.fft.rfftfreq(n_fft_padded, 1/fs)
        # Using a window function (like Hanning) helps suppress sidelobes in noisy data
        window = np.hanning(len(if_output_mag))
        if_output_mag_padded = np.pad(if_output_mag * window, (0, n_fft_padded - len(if_output_mag)), mode='constant')
        fft_magnitude = np.abs(np.fft.rfft(if_output_mag_padded))

        
        # Plot the IF output (magnitude) and I/Q traces in real time
        if run_plot:
            if plot_fig is None:
                plot_fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            else:
                for ax in axs.flat:
                    ax.clear()

            axs[0, 0].plot(real, label='Transmitted I')
            axs[0, 0].plot(imag, label='Transmitted Q')
            axs[0, 0].set_title(f'Run {run_index+1} - Transmitted I/Q vs Sample')
            axs[0, 0].set_xlabel('Sample')
            axs[0, 0].set_ylabel('Amplitude')
            axs[0, 0].legend()
            axs[0, 0].grid(True)

            axs[0, 1].specgram(real, Fs=fs)
            axs[0, 1].set_title(f'Run {run_index+1} - Transmitted I Spectrogram')
            axs[0, 1].set_ylabel('Frequency')

            axs[1, 0].plot(rx_real, label='Received I')
            axs[1, 0].plot(rx_imag, label='Received Q')
            axs[1, 0].set_title(f'Run {run_index+1} - Received I/Q vs Sample')
            axs[1, 0].set_xlabel('Sample')
            axs[1, 0].set_ylabel('Amplitude')
            axs[1, 0].legend()
            axs[1, 0].grid(True)

            axs[1, 1].specgram(rx_real, Fs=fs)
            axs[1, 1].set_title(f'Run {run_index+1} - Received I Spectrogram')
            axs[1, 1].set_ylabel('Frequency')

            plot_fig.canvas.draw()
            plot_fig.canvas.flush_events()
            plt.pause(0.01)
            if not plt.fignum_exists(plot_fig.number):
                print('Plot window closed, stopping realtime loop.')
                break

        bw = bandwidth_cycles_per_sample * fs
        print(f"Bandwidth (Hz): {bw}")
        duration = sweep_duration_samples / fs
        print(f"Sweep Duration (s): {duration}")
        Slope = bw / duration
        c = 299792458
        tau = if_average / Slope
        distance = tau * c / 2
        print(f"t={tau},distance={distance}")
        range_axis = (f_axis * c * duration) / (2 * bw)


        if plot_fig is None:
                plot_fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        else:
            for ax in axs.flat:
                ax.clear()
        
        axs[0, 0].plot(if_output, label='IF output')
        axs[0, 0].set_title(f'Run {run_index+1} - IF output')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(range_axis, 20 * np.log10(fft_magnitude), label='IF output')
        axs[0, 1].set_title(f'Run {run_index+1} - verage Range Profile (matched filter via FFT)')
        axs[0, 1].set_xlabel('Range (m)')
        # axs[0, 1].set_xlim(0, 20)
        # axs[0, 1].set_ylim(170, 240)
        axs[0, 1].set_ylabel('Amplitude (dB)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(real, label='TX I')
        axs[1, 0].plot(rx_real, label='RX I')
        axs[1, 0].set_title(f'Run {run_index+1} - TX vs RX I')
        axs[1, 0].set_xlabel('Sample')
        axs[1, 0].set_ylabel('Amplitude')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].specgram(rx_real, Fs=fs)
        axs[1, 1].set_title(f'Run {run_index+1} - Received I Spectrogram')
        axs[1, 1].set_ylabel('Frequency')

        plot_fig.canvas.draw()
        plot_fig.canvas.flush_events()

        plt.pause(0.001)
        run_index += 1

except KeyboardInterrupt:
    print('Realtime plot stopped by user.')

if run_plot and plot_fig is not None:
    plt.ioff()
    plt.show(block=True)

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

