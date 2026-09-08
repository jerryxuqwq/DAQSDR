# read_binary_data.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import fftconvolve


# --- Parameters ---
RX_FILE = "rx_iq.bin"
TX_FILE = "tx_iq.bin"
NUM_SAMPLES = int(1024*1024/8)   # Must match the number of samples saved in the original script
SAMPLE_RATE = 1e9    # Must match the sample rate used

# --- SPECTROGRAM for RX ---
WIN_LEN = 128
win = signal.windows.hann(WIN_LEN)

HOP = (16)#125#256
SFT = signal.ShortTimeFFT(
    win,
    hop=HOP,
    fs=SAMPLE_RATE,
    fft_mode="centered",
    scale_to="psd",
)

# Read the complex data from the binary files
# The dtype=np.complex128 is crucial as it matches how the file was saved
try:
    rx_iq = np.fromfile(RX_FILE, dtype=np.complex128, count=NUM_SAMPLES)
    tx_iq_ref = np.fromfile(TX_FILE, dtype=np.complex128, count=NUM_SAMPLES)
    #rx_iq=rx_iq[100000:NUM_SAMPLES]  # Ensure we only take the expected number of samples
    #tx_iq_ref=tx_iq_ref[100000:NUM_SAMPLES]  # Ensure we
    
    if rx_iq.size == 0:
        print(f"Error: Could not read data from {RX_FILE}. File might be empty.")
    elif tx_iq_ref.size == 0:
        print(f"Error: Could not read data from {TX_FILE}. File might be empty.")
    else:
        print(f"Read {len(rx_iq)} complex samples from {RX_FILE}")
        print(f"Read {len(tx_iq_ref)} complex samples from {TX_FILE}")

        # --- Matched Filter: Mix RX with TX (complex multiplication) ---
        if_output = rx_iq * np.conj(tx_iq_ref[::-1])

        #TX_matched = np.conjugate(tx_iq_ref[::-1])
        #if_output = fftconvolve(rx_iq, TX_matched, mode='same')

        if_output = if_output[int(4e-6*SAMPLE_RATE):NUM_SAMPLES]  # Ensure we only take the expected number of samples for plotting

        Sxx_if = SFT.spectrogram(if_output)
        Sxx_dB_if = 10 * np.log10(np.maximum(Sxx_if, 1e-12))

        t = np.arange(Sxx_if.shape[1]) * SFT.delta_t
        f = SFT.f

        # plt.figure(figsize=(12, 6))
        # plt.pcolormesh(t, f / 1e6, Sxx_dB_if, shading="auto")
        # plt.colorbar(label="Power [dB]")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Frequency [MHz]")
        # plt.tight_layout()
        # plt.show()
        
        if_average = np.average(np.abs(if_output))
        start_freq =-200e6
        end_freq = 200e6
        duration = NUM_SAMPLES / SAMPLE_RATE
        print(f"if_average={if_average},start_freq={start_freq},end_freq={end_freq},duration={duration}")

        Slope = (end_freq-start_freq)/duration
        c=299792458
        tau =if_average/Slope
        distance =tau  *c/2
        print(f"t={tau},distance={distance}")

        # Plot the IF output (magnitude)
        # plt.figure(figsize=(12, 6))
        # plt.plot(if_output)
        # plt.title("IF Output (Magnitude) after Mixing RX with TX")
        # plt.xlabel("Sample Number")
        # plt.ylabel("Magnitude")
        # plt.grid(True)
        # plt.show()




        Sxx_rx = SFT.spectrogram(rx_iq)
        Sxx_dB_rx = 10 * np.log10(np.maximum(Sxx_rx, 1e-12))

        t = np.arange(Sxx_rx.shape[1]) * SFT.delta_t
        f = SFT.f

        # --- SPECTROGRAM for TX ---
        Sxx_tx = SFT.spectrogram(tx_iq_ref)
        Sxx_dB_tx = 10 * np.log10(np.maximum(Sxx_tx, 1e-12))

        t = np.arange(Sxx_tx.shape[1]) * SFT.delta_t
        f = SFT.f

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

        # RX
        im_rx = ax.pcolormesh(t, f / 1e6, Sxx_dB_rx, shading="auto", cmap="viridis")
        cbar = fig.colorbar(im_rx, ax=ax, label="Power [dB] (RX)")

        # TX overlaid (transparent)
        im_tx = ax.pcolormesh(
            t, f / 1e6, Sxx_dB_tx,
            shading="auto",
            cmap="magma",
            alpha=0.45
        )
        # optional second colorbar (comment out if you don't want two)
        fig.colorbar(im_tx, ax=ax, label="Power [dB] (TX)")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [MHz]")
        ax.set_title("RX (viridis) overlaid with TX (magma) spectrogram")

        plt.show()





except FileNotFoundError:
    print(f"Error: The file was not found.")
