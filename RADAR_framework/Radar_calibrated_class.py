import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt

import adi

# Custom local modules (assumed to exist in your directory)
import WaveformGenerator
import calibration
from MatchFilter import matched_filter
from Radar_calibration_sim import FMCWRadar


class FMCWRadarProcessor:
    """
    A class to handle configuration, transmission, reception, and processing 
    of FMCW radar signals using an Analog Devices DAQ2 board.
    """
    
    def __init__(self, ip_address="ip:analog.local", num_samples=int(1024*1024/4/4)):
        self.ip_address = ip_address
        self.num_samples = num_samples
        self.amplitude = 1.0
        
        # Device and waveform parameters
        self.dev = None
        self.fs = None
        self.generator = None
        
        # Signal data
        self.tx_iq = None
        self.tx_iq_ref = None
        self.rx_iq = None
        self.rx_real = None
        self.rx_imag = None
        
        # Processing results
        self.if_output = None
        self.fft_magnitude = None
        self.f_axis = None
        self.range_axis = None
        self.bw = None
        self.duration = None
        self.chirp_rate = None

        # Speed of light
        self.c = 299792458

    def initialize_device(self):
        """Connects to and configures the ADI DAQ2 device."""
        print(f"Connecting to device at {self.ip_address}...")
        self.dev = adi.DAQ2(self.ip_address)
        self.dev._ctx.set_timeout(3000)
        self.dev._rxadc.set_kernel_buffers_count(1)
        self.dev._txdac.set_kernel_buffers_count(2)
        
        self.dev.rx_enabled_channels = [0, 1]
        self.dev.tx_enabled_channels = [0, 1]
        self.dev.rx_buffer_size = self.num_samples
        self.dev.tx_cyclic_buffer = False
        
        self.fs = int(self.dev.sample_rate)
        self.dev.dds_single_tone(self.fs / 10, 0.0, channel=0)
        print("Device Started.")

    def generate_waveform(self, f_start_cycles=0.1, bw_cycles=0.2):
        """Generates the FMCW waveform and computes transmission reference."""
        print("Generating Waveform...")
        self.generator = WaveformGenerator.WaveformGenerator(
            num_samples=int(self.num_samples / 2), 
            fs=self.fs, 
            amplitude=2**13
        )

        sweep_duration_samples = int(self.num_samples / 2)
        
        _, fmcw_signal = self.generator.generate_fmcw(
            f_start_cycles_per_sample=f_start_cycles,  
            sweep_duration_samples=sweep_duration_samples,  
            bandwidth_cycles_per_sample=bw_cycles 
        )

        # Calculate parameters for later processing
        self.chirp_rate = bw_cycles / sweep_duration_samples
        self.bw = bw_cycles * self.fs
        self.duration = sweep_duration_samples / self.fs
        
        print(f"Chirp Rate (Hz/s): {self.chirp_rate}")
        print(f"Bandwidth (Hz): {self.bw}")
        print(f"Sweep Duration (s): {self.duration}")

        self.tx_real = np.real(fmcw_signal)
        self.tx_imag = np.imag(fmcw_signal)
        self.tx_iq = [self.tx_real, self.tx_imag]
        self.tx_iq_ref = self.tx_iq[0] + 1j * self.tx_iq[1]

    def capture_data(self, runs=1):
        """Arms the device, transmits the signal, and captures the received data."""
        for r in range(runs):
            start = time.perf_counter()
            self.dev.rx_sync_start = "arm"
            self.dev.tx_sync_start = "arm"

            if not ("arm" == self.dev.tx_sync_start == self.dev.rx_sync_start):
                raise Exception(
                    f"Unexpected SYNC status: TX {self.dev.tx_sync_start} RX: {self.dev.rx_sync_start}"
                )
            
            if r == 0:
                self.dev.tx_destroy_buffer()
                
            self.dev.rx_destroy_buffer()
            self.dev._rx_init_channels()
            self.dev.tx(self.tx_iq)
            self.dev.tx_sync_start = "trigger_manual"

            if not ("disarm" == self.dev.tx_sync_start == self.dev.rx_sync_start):
                raise Exception(
                    f"Unexpected SYNC status: TX {self.dev.tx_sync_start} RX: {self.dev.rx_sync_start}"
                )

            try:
                x = self.dev.rx()
            except Exception as e:
                print(f"Run #{r} FAILED: {e}")
                continue
            
            end = time.perf_counter()
            print(f"Run {r+1} completed in {end - start:.4f} seconds")

            # Reconstruct complex RX signal
            self.rx_real = x[0]
            self.rx_imag = x[1]
            self.rx_iq = self.rx_real + 1j * self.rx_imag

    def process_signal(self, cutoff_freq=300E6):
        """Performs matched filtering, LPF, and FFT processing."""
        # 1. Matched Filtering (Element-wise multiplication with complex conjugate)
        tx_iq_ref_padded = np.pad(
            self.tx_iq_ref[::-1], 
            (0, max(0, len(self.rx_iq) - len(self.tx_iq_ref))), 
            mode='constant'
        )
        self.if_output = self.rx_iq * np.conj(tx_iq_ref_padded)
        if_output_og = copy.deepcopy(self.if_output)

        # 2. Apply Low Pass Filter
        normalized_cutoff = cutoff_freq / self.fs
        b, a = butter(4, normalized_cutoff, btype='low')
        self.if_output = filtfilt(b, a, self.if_output)

        # 3. FFT Processing Setup
        n_fft = len(self.if_output)
        n_fft_padded = n_fft  

        self.f_axis = np.fft.fftfreq(n_fft_padded, 1/self.fs)
        window = np.hanning(n_fft)
        if_output_windowed = self.if_output * window

        # 4. Zero Padding and FFT
        if_output_padded = np.pad(if_output_windowed, (0, n_fft_padded - n_fft), mode='constant')
        fft_result = np.fft.fft(if_output_padded)
        self.fft_magnitude = np.abs(fft_result)

        # 5. Range Calculation
        if_average = 0
        slope = self.bw / self.duration
        tau = if_average / slope
        distance = tau * self.c / 2
        print(f"t = {tau}, distance = {distance}")

        self.range_axis = (self.f_axis * self.c * self.duration) / (2 * self.bw)

    def plot_results(self, show_debug_plots=False):
        """Visualizes the processed data."""
        # Main Dashboard Plot
        plot_fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        
        # Plot IF Output
        axs[0, 0].plot(np.real(self.if_output), label='IF output (real)', alpha=0.5)
        axs[0, 0].set_title('IF output')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot Average Range Profile
        axs[0, 1].plot(self.range_axis, 20 * np.log10(self.fft_magnitude + 1e-12), label='IF output')
        axs[0, 1].set_title('Average Range Profile (matched filter via FFT)')
        axs[0, 1].set_xlabel('Range (m)')
        axs[0, 1].set_ylabel('Amplitude (dB)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot TX vs RX (I channel)
        axs[1, 0].plot(self.tx_real, label='TX I')
        axs[1, 0].plot(self.rx_real, label='RX I')
        axs[1, 0].set_title('TX vs RX I')
        axs[1, 0].set_ylabel('Amplitude')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Spectrogram
        axs[1, 1].specgram(self.rx_iq, Fs=self.fs)
        axs[1, 1].set_title('Received I Spectrogram')
        axs[1, 1].set_ylabel('Frequency')
        
        plt.show()

        # Optional debug plots
        if show_debug_plots:
            self._plot_debug()

    def _plot_debug(self):
        """Secondary plot view equivalent to original run_plot=True state."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.tx_real, label='Transmitted I')
        plt.plot(self.tx_imag, label='Transmitted Q')
        plt.title('Transmitted I/Q vs Sample')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.specgram(self.tx_real, Fs=self.fs, label='Transmitted I Spectrogram')
        plt.title('Transmitted I Spectrogram')
        plt.ylabel('Frequency')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.plot(self.rx_real, label='Received I')
        plt.plot(self.rx_imag, label='Received Q')
        plt.title('Received I/Q vs Sample')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.specgram(self.rx_real, Fs=self.fs, label='Received I Spectrogram')
        plt.title('Received I Spectrogram')
        plt.ylabel('Frequency')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

    def run_pipeline(self, show_debug_plots=False):
        """Convenience method to execute the standard end-to-end flow."""
        self.initialize_device()
        self.generate_waveform()
        self.capture_data(runs=1)
        self.process_signal()
        self.plot_results(show_debug_plots=show_debug_plots)


# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Initialize the class
    radar = FMCWRadarProcessor(ip_address="ip:analog.local")
    
    # Run the full pipeline (Set show_debug_plots=True to view the secondary charts)
    radar.run_pipeline(show_debug_plots=False)