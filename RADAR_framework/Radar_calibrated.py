import adi
import time
import copy

from scipy import signal
import calibration
import WaveformGenerator

import matplotlib.pyplot as plt
import numpy as np

from MatchFilter import matched_filter
import pickle
from scipy.signal import butter, filtfilt


from Radar_calibration_sim import FMCWRadar

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
bandwidth_cycles_per_sample=0.2  # Normalized bandwidth (5 MHz / fs)

FMCW = generator.generate_fmcw(f_start_cycles_per_sample=f_start_cycles_per_sample,  
                               sweep_duration_samples=sweep_duration_samples,  
                               bandwidth_cycles_per_sample=bandwidth_cycles_per_sample 
                               )[1]

# Calculate chirp rate
chirp_rate = (bandwidth_cycles_per_sample) / (sweep_duration_samples)
print(f"Chirp Rate (Hz/s): {chirp_rate}")

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

for r in range(1):
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
    if r == 0:
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
        print("Run#", r, " ----------------------------- FAILED:", e)
        continue
    end = time.perf_counter()

    print(f"Run {r+1} completed in {end - start:.4f} seconds")
    # 1. Reconstruct complex RX signal
    rx_real = x[0] * 2
    rx_imag = x[1] * 2
    rx_iq = rx_real + 1j * rx_imag

    # 2. Matched Filtering (Cross-correlation)
    # Using method='fft' performs the multiplication in the frequency domain automatically.
    # signal.correlate natively handles the complex conjugation and flipping required for matched filtering.
    #if_output = signal.correlate(rx_iq, tx_iq_ref, mode='full', method='fft')
    #tx_iq_ref_padded = np.pad(tx_iq_ref[::-1], (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
    tx_iq_ref_padded = np.pad(tx_iq_ref, (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
    if_output = rx_iq * np.conj(tx_iq_ref_padded)  # Element-wise multiplication with complex conjugate of reference (matched filter)
    if_output_og=copy.deepcopy(if_output)
    with open('rx_calibration.pkl', 'wb') as f:
        pickle.dump(if_output, f)
    print("RX calibration data saved to rx_calibration.pkl")
   

    # Apply low pass filter to if_output
    # cutoff_freq = 300E6 # Cutoff at half the bandwidth
    # normalized_cutoff = cutoff_freq / (fs /2)
    # b, a = butter(4, normalized_cutoff, btype='low')
    # if_output = filtfilt(b, a, if_output)

    #Calibrate the IF output using the calibration function
    t = np.arange(len(if_output)) / fs  # Time vector corresponding to the IF output samples
    chirp_rate_hz_per_s = (bandwidth_cycles_per_sample * fs) / (sweep_duration_samples / fs)

    if_output = calibration.calibration_from_range(if_output, t,chirp_rate_hz_per_s, 0.1) #TODO
    # 4. FFT Processing Setup
    n_fft = len(if_output)
    window = np.hanning(n_fft)
    fft_result = np.fft.fftshift(np.fft.fft(if_output * window))
    f_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, 1/fs))
    fft_magnitude = np.abs(fft_result)
    # # Optional but recommended: Shift the FFT so 0 Hz (DC) is in the center of your plot
    # f_axis = np.fft.fftshift(f_axis)
    # fft_magnitude = np.fft.fftshift(fft_magnitude)

    if_average = 0
    # #Plot the IF output (magnitude)
    # plt.figure(figsize=(12, 6))
    # plt.plot(if_output)
    # plt.title("IF Output (Magnitude) after Mixing RX with TX")
    # plt.xlabel("Sample Number")
    # plt.ylabel("Magnitude")
    # plt.grid(True)
    # plt.show()
    bw= bandwidth_cycles_per_sample*fs
    print(f"Bandwidth (Hz): {bw}")
    duration = sweep_duration_samples/fs
    print(f"Sweep Duration (s): {duration}")
    Slope = bw/duration
    c=299792458
    tau =if_average/Slope
    distance =tau  *c/2
    print(f"t={tau},distance={distance}")

    plot_fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    
    axs[0, 0].plot(if_output, label='IF output',alpha=0.5)
    #axs[0, 0].plot(if_output_og, label='IF output_OG',alpha=0.5)

    axs[0, 0].set_title(f'IF output')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    slope = (bandwidth_cycles_per_sample * fs) / (sweep_duration_samples / fs)  # Hz/s
    range_axis = f_axis * c / (2 * slope)
    axs[0, 1].plot(range_axis, 20 * np.log10(fft_magnitude), label='IF output')
    print(fft_magnitude)
    
    # Find and display all peaks
    #peaks, _ = signal.find_peaks(20 * np.log10(fft_magnitude), height=-50)
    #print(peaks)
    #axs[0, 1].plot(range_axis[peaks], 20 * np.log10(fft_magnitude)[peaks], "x", color='red', label='Peaks')
    axs[0, 1].set_title(f'Average Range Profile (matched filter via FFT)')
    axs[0, 1].set_xlabel('Range (m)')
    #axs[0, 1].set_xlim(0, 100)
    axs[0, 1].autoscale(enable=True, axis='x', tight=True)
    axs[0, 1].set_ylabel('Amplitude (dB)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(real, label='TX I')
    axs[1, 0].plot(rx_real, label='RX I')
    axs[1, 0].set_title(f'TX vs RX I')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].specgram(rx_iq, Fs=fs)
    axs[1, 1].set_title(f'Received I Spectrogram')
    axs[1, 1].set_ylabel('Frequency')
    plt.show()




    if run_plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(real, label='Transmitted I')
        plt.plot(imag, label='Transmitted Q')
        plt.title(f'Run {r+1} - Transmitted I/Q vs Sample')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.specgram(real, Fs=fs, label='Transmitted I Spectrogram')
        plt.title(f'Run {r+1} - Transmitted I Spectrogram')
        plt.ylabel('Frequency')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.plot(rx_real, label='Received I')
        plt.plot(rx_imag, label='Received Q')
        plt.title(f'Run {r+1} - Received I/Q vs Sample')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.specgram(rx_real, Fs=fs, label='Received I Spectrogram')
        plt.title(f'Run {r+1} - Received I Spectrogram')
        plt.ylabel('Frequency')
        plt.colorbar()
        plt.show()
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

