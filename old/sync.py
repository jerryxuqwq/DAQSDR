import adi
import numpy as np

# --- Configuration Parameters ---

# Use the 'local' context when running the script directly on the board
DEVICE_URI = "ip:analog.local" 

# Hardware parameters
SAMPLE_RATE = int(1e9)  # 1 GSPS for AD9680 (RX) and AD9144 (TX)


# Signal parameters
NUM_SAMPLES = int(1024*1024*1/4)   # Total number of samples to transmit and receive (must match the number saved in the original script)
CHIRP_FREQ_START = -200e6 # Chirp start frequency relative to center
CHIRP_FREQ_STOP = 200e6   # Chirp stop frequency relative to center

# File to save the received data
# --- Main Program ---

import numpy as np

def generate_chirp(sample_rate, num_samples, f_start, f_stop, length):
    """
    Generates a complex digital chirp signal padded to num_samples.
    Chirp occupies `length` samples centered (as close as possible) with zeros front/back.
    Returns separate I and Q arrays of type int16.
    """
    if length > num_samples:
        raise ValueError("Chirp length cannot exceed total number of samples.")
    
    # How much to pad on each side
    pad_before = 0#(num_samples - length) // 2
    pad_after = num_samples - length - pad_before

    # Generate chirp segment
    t = np.arange(length) / sample_rate
    k = (f_stop - f_start) / (length / sample_rate)
    phase = 2 * np.pi * (f_start * t + (k / 2) * t**2)
    iq_data = np.exp(1j * phase) * (2**15 - 1)

    tx_i_chirp = np.real(iq_data).astype(np.int16)
    tx_q_chirp = np.imag(iq_data).astype(np.int16)

    # Pad zeros
    tx_i = np.pad(tx_i_chirp, (pad_before, pad_after), mode='constant')
    tx_q = np.pad(tx_q_chirp, (pad_before, pad_after), mode='constant')

    return tx_i, tx_q

def main():
    """Main function to run the FMCDAQ2 chirp test using explicit DMA control."""
    
    # 1. Generate the chirp data as separate I and Q arrays
    tx_i_data, tx_q_data = generate_chirp(
        SAMPLE_RATE, NUM_SAMPLES, CHIRP_FREQ_START, CHIRP_FREQ_STOP, length=int(1024)
    )
    tx_data_list = [tx_i_data, tx_q_data] # A , B

    sdr = None

    try:
        # 2. Initialize the FMCDAQ2 board
        print(f"Connecting to DAQ2 using '{DEVICE_URI}' context...")
        sdr = adi.DAQ2(uri=DEVICE_URI)
        print("Connection successful.")
        
        # 3. Configure hardware settings
        print("Configuring hardware...")
        #sdr.rx_sample_rate = SAMPLE_RATE
        #sdr.tx_sample_rate = SAMPLE_RATE
        sdr._ctx.set_timeout(3000)
        
        sdr.rx_enabled_channels = [0, 1] # I0, Q0
        sdr.tx_enabled_channels = [0, 1] # I0, Q0
        sdr.rx_buffer_size = NUM_SAMPLES
        sdr.dds_enabled = [False] * 2
        # Set TX to be cyclic for continuous transmission
        sdr.tx_cyclic_buffer = False
        
        sdr.rx_sync_start = "arm"
        sdr.tx_sync_start = "arm"
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        sdr._rx_init_channels()

        # 4. Perform DMA Sync using explicit start/wait/stop calls

        # Enable (arm) the DMA engines
        if not ("arm" == sdr.tx_sync_start == sdr.rx_sync_start):
            raise Exception(
                "Unexpected SYNC status: TX "
                + sdr.tx_sync_start
                + " RX: "
                + sdr.rx_sync_start
            )
        
        # Start the DMAs. It's good practice to start RX first.
        print("Starting DMA sync using explicit control...")
        sdr.tx(tx_data_list)
        #input("Press Enter to continue...")
        sdr.tx_sync_start = "trigger_manual"
        

        if not ("disarm" == sdr.tx_sync_start == sdr.rx_sync_start):
            raise Exception(
                "Unexpected SYNC status: TX "
                + sdr.tx_sync_start
                + " RX: "
                + sdr.rx_sync_start
        )        
        # Retrieve the data that was captured into the buffer
        rx_data_list = sdr.rx()
        print("DMA transfer complete.")
        
        # 5. Process and Save Received Data
        print(f"Processing and saving received data ...")
        
        rx_i_data = np.real(rx_data_list[0]) # A
        rx_q_data = np.imag(rx_data_list[1]) # B

        rx_iq_data = rx_i_data + 1j * rx_q_data
        tx_iq_data = tx_i_data + 1j * tx_q_data
        

        # Save the raw complex data to a binary file
        rx_iq_data.tofile("rx_iq.bin")
        tx_iq_data.tofile("tx_iq.bin")
        print(f"Successfully saved {len(rx_iq_data)} complex samples to rx_iq.bin.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the SDR object to release hardware resources
        if sdr:
            del sdr
            print("SDR object cleaned up.")

if __name__ == "__main__":
    main()
    # read_binary_data.py
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy.signal import fftconvolve


    # --- Parameters ---
    RX_FILE = "rx_iq.bin"
    TX_FILE = "tx_iq.bin"
    #NUM_SAMPLES = int(1024*1024/8)   # Must match the number of samples saved in the original script
    SAMPLE_RATE = 1e9    # Must match the sample rate used

    # --- SPECTROGRAM for RX ---
    WIN_LEN = 1024
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

        # Plot RX and TX IQ data
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(np.real(rx_iq), label='RX I', alpha=0.7)
        plt.plot(np.imag(rx_iq), label='RX Q', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Received IQ Data')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.real(tx_iq_ref), label='TX I', alpha=0.7)
        plt.plot(np.imag(tx_iq_ref), label='TX Q', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Transmitted IQ Data')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
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
