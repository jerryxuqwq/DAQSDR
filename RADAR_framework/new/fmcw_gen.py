import adi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import butter, filtfilt, chirp
import pickle

class FMCW_Radar:
    def __init__(self, uri="ip:analog.local",
                 num_samples=int(1024 * 1024 / 4 / 4), 
                 timeout=3000,
                 fs=1e9, 
                 f_start=-500e6,
                 f_stop=500e6,
                 tx_duration=1,
                 R_cal=0.1):  # Calibration range (m)
        
        self.fs = fs
        self.f_start = f_start
        self.f_stop = f_stop
        self.duration = num_samples / fs
        self.tx_duration = tx_duration
        self.c = 3e8  # Speed of light
        self.num_samples = num_samples
        self.R_cal = R_cal
        
        self.dev = adi.DAQ2(uri)
        self.dev._ctx.set_timeout(timeout)
        self.dev._rxadc.set_kernel_buffers_count(1)
        self.dev._txdac.set_kernel_buffers_count(2)
        self.dev.rx_enabled_channels = [0, 1]
        self.dev.tx_enabled_channels = [0, 1]
        self.dev.rx_buffer_size = num_samples
        self.dev.tx_cyclic_buffer = False
        
        self.fs = int(self.dev.sample_rate)
        self.dev.dds_single_tone(self.fs / 10, 0.0, channel=0)
        self._first_run = True
        self.chirp_rate = (f_stop - f_start) / tx_duration * self.duration
        print("Device Started")

    def ota(self, tx_data_complex):
        tx_data = [np.real(tx_data_complex), np.imag(tx_data_complex)]
        dev = self.dev
        dev.rx_sync_start = "arm"
        dev.tx_sync_start = "arm"
        
        if not ("arm" == dev.tx_sync_start == dev.rx_sync_start):
            raise Exception(
                "Unexpected SYNC status: TX "
                + dev.tx_sync_start
                + " RX: "
                + dev.rx_sync_start
            )

        if self._first_run:
            dev.tx_destroy_buffer()
            self._first_run = False
            
        dev.rx_destroy_buffer()
        dev._rx_init_channels()
        dev.tx(tx_data)
        dev.tx_sync_start = "trigger_manual"

        if not ("disarm" == dev.tx_sync_start == dev.rx_sync_start):
            raise Exception(
                "Unexpected SYNC status: TX "
                + dev.tx_sync_start
                + " RX: "
                + dev.rx_sync_start
            )

        rx_data = dev.rx()
        return rx_data[0] * 11 + 1j * rx_data[1] * 11
    def generate_swept_noise(self,noisetype, bandwidth, f_start, f_end, fs, duration=1.0):
        """
        Generates noise with a constant bandwidth but a center frequency that 
        sweeps from f_start to f_end over the given duration.
        
        Parameters:
        * noisetype (str): 'gaussian' or 'uniform'.
        * bandwidth (float): The width of the frequency band in Hz.
        * f_start (float): Starting center frequency in Hz.
        * f_end (float): Ending center frequency in Hz.
        * fs (int): The sampling frequency in Hz.
        * duration (float): Length of the noise in seconds.
        
        Returns:
        * t (numpy array): Time vector.
        * swept_noise (numpy array): The frequency-swept noise signal.
        """
        
        # 1. Calculate time vector
        num_samples = int(fs * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # 2. Generate raw white noise
        if noisetype.lower() == 'gaussian':
            raw_noise = np.random.normal(0, 1, num_samples)
        elif noisetype.lower() == 'uniform':
            raw_noise = np.random.uniform(-1, 1, num_samples)
        else:
            raise ValueError("Unsupported noisetype. Please use 'gaussian' or 'uniform'.")
            
        # 3. Create Baseband Noise (Low-pass filter at Bandwidth / 2)
        # Multiplying baseband noise by a carrier splits the bandwidth symmetrically,
        # so we only need to filter at half the desired total bandwidth.
        nyquist = 0.5 * fs
        lowpass_cutoff = bandwidth / 2.0
        
        if lowpass_cutoff <= 0 or lowpass_cutoff >= nyquist:
            raise ValueError("Invalid bandwidth for the given sampling frequency.")
            
        b, a = butter(4, lowpass_cutoff / nyquist, btype='lowpass')
        baseband_noise = filtfilt(b, a, raw_noise *2**14 )
        
        # 4. Generate the FMCW Carrier (Chirp)
        # This acts as our moving center frequency
        carrier = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
        
        # 5. Modulate!
        # Multiplying the baseband noise by the carrier shifts the noise up to the carrier's frequency.
        swept_noise = baseband_noise * carrier
        
        return t, swept_noise

    def generate_fmcw_chirp(self):
        t_whole = np.arange(0, self.duration, 1 / self.fs)
        t = np.arange(0, self.duration * self.tx_duration, 1 / self.fs)
        k = (self.f_stop - self.f_start) / self.duration  # Chirp rate
        chirp_signal = np.exp(1j * 2 * np.pi * (self.f_start * t + 0.5 * k * t**2)) * 2**10
        
        # pad the chirp signal to match the total number of samples
        if len(chirp_signal) < len(t_whole):
            chirp_signal = np.pad(chirp_signal, (0, len(t_whole) - len(chirp_signal)), mode='constant')

        return t_whole, chirp_signal, t

    def generate_cw_waveform(self, freq, duration=None, amplitude=1.0, phase=0.0, num_samples=None):
        """
        Generate a continuous-wave (CW) complex baseband waveform.

        Parameters
        - freq: carrier frequency in Hz (relative to baseband)
        - duration: length in seconds (optional if num_samples provided)
        - amplitude: linear amplitude (default 1.0)
        - phase: initial phase in radians (default 0.0)
        - num_samples: explicit number of samples to generate (overrides duration)

        Returns (t, cw_signal) where cw_signal is a complex numpy array.
        """
        fs = getattr(self, 'fs', None) or 1.0

        if num_samples is None:
            if duration is None:
                num_samples = int(getattr(self, 'num_samples', 0))
            else:
                num_samples = int(np.round(duration * fs))

        t = np.arange(num_samples) / fs
        cw_signal = amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))
        return t, cw_signal
    
    def matched_filter(self, rx_iq, tx_iq_ref):
        # Matched filtering (cross-correlation)
        if_output = rx_iq * np.conj(tx_iq_ref)  # Element-wise multiplication with complex conjugate of reference
        return if_output
    
    def low_pass_filter(self, signal, cutoff_freq):
        normalized_cutoff = cutoff_freq / (self.fs / 2)
        b, a = butter(10, normalized_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def range_fft(self, if_output, n_fft=None):
        """Compute the range profile via FFT of the IF (beat) signal."""
        n = len(if_output)
        if n_fft is None:
            n_fft = n
        
        # Apply window to reduce sidelobes
        window = np.hanning(n)
        windowed = if_output * window
        
        # Zero pad
        if n_fft > n:
            windowed = np.pad(windowed, (0, n_fft - n), mode='constant')
        
        fft_result = np.fft.fft(windowed, n=n_fft)
        fft_mag = np.abs(fft_result)
        
        # Frequency axis (complex signal -> use full fftfreq)
        f_axis = np.fft.fftfreq(n_fft, 1 / self.fs)
        
        # Shift to center
        fft_mag = np.fft.fftshift(fft_mag)
        f_axis = np.fft.fftshift(f_axis)
        
        return f_axis, fft_mag
    
    def save_calibration(self, filename, calibration_data):
        with open(filename, 'wb') as f:
            pickle.dump(calibration_data, f)

    def load_calibration(self, filename):
        with open(filename, 'rb') as f:
            calibration_data = pickle.load(f)
        return calibration_data
    
    def calibrate_from_range(self, if_output, t, chirp_rate, calibration_data):
        W = chirp_rate
        R_cal = self.R_cal
        c = self.c
        
        phase_term = np.exp(4j * np.pi * W * R_cal * t / c)
        calibrated_if = (if_output / calibration_data) * phase_term
        
        return calibrated_if

def main():
    num_samples = int(1024 * 1024 / 4 / 4)

    radar = FMCW_Radar(
        num_samples=num_samples,
        f_start=-400e6,
        f_stop=400e6,
        tx_duration=0.5
        )
    
    time, tx_chirp_complex, tx_time = radar.generate_fmcw_chirp()
    
    freq = -470e6
    # generate CW for 10% of the total samples (make sure it's an int)
    cw_samples = int(num_samples * 0.1)
    time, cw = radar.generate_cw_waveform(freq=freq, num_samples=cw_samples, amplitude=2**10)

    # Zero-pad the shorter signal so we can add them element-wise
    len_tx = len(tx_chirp_complex)
    len_cw = len(cw)
    if len_cw < len_tx:
        cw = np.pad(cw, (0, len_tx - len_cw), mode='constant')
    elif len_tx < len_cw:
        tx_chirp_complex = np.pad(tx_chirp_complex, (0, len_cw - len_tx), mode='constant')

    total = cw + tx_chirp_complex
    rx_iq = radar.ota(total)

    plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.size': 16
})

    plt.figure(figsize=(7.5, 4.5)) 

    # --- ADDED: 0 dB Peak Normalization ---
    # 1. Calculate the raw power (Pxx) without plotting
    Pxx, _, _ = mlab.specgram(rx_iq, Fs=1000e6)

    # 2. Scale the input signal so the maximum Pxx becomes exactly 1 (0 dB)
    rx_iq_norm = rx_iq / np.sqrt(np.max(Pxx))
    # --------------------------------------

    # Plot the 0 dB normalized signal
    plt.specgram(rx_iq_norm, Fs=1000e6, cmap='viridis')

    plt.xlabel('Time (µs)')
    plt.ylabel('Frequency (GHz)')
    #plt.title('Spectrogram of RX IQ Signal')

    plt.colorbar(label='Normalized Amplitude (dB)')

    # --- ADDED: Lock the color range ---
    # Forces the colorbar to strictly range from -80 dB to 0 dB. 
    # Adjust -80 to a different noise floor (like -60 or -100) if needed.
    plt.clim(vmin=-80, vmax=0) 

    # Convert axes to µs and GHz
    ax = plt.gca()
    plt.draw() 

    ax_time = ax.get_xticks() * 1e6  # Convert to µs
    ax.set_xticklabels([f'{t:.2f}' for t in ax_time])

    ax_freq = ax.get_yticks() / 1e9  # Convert to GHz
    ax.set_yticklabels([f'{f:.2f}' for f in ax_freq])

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()