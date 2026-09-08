import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt


# =====================================================
# FMCW Radar Simulation Parameters
# =====================================================
class FMCWRadar:
    def __init__(self):
        # System parameters
        self.fs = 1e9                    # Sample rate (1 GHz)
        self.c = 299792458               # Speed of light (m/s)
        
        # FMCW chirp parameters
        self.f_start = 100e6             # Start frequency (100 MHz, baseband)
        self.bandwidth = 200e6           # Sweep bandwidth (200 MHz)
        self.sweep_duration = 65536 / self.fs  # Sweep duration (s)
        self.num_samples = int(self.sweep_duration * self.fs)
        
        # Derived parameters
        self.chirp_rate = self.bandwidth / self.sweep_duration  # Hz/s
        self.range_resolution = self.c / (2 * self.bandwidth)
        self.max_range = (self.fs / 2) * self.c / (2 * self.chirp_rate)
        
        print(f"=== FMCW Radar Configuration ===")
        print(f"Sample rate:         {self.fs/1e6:.2f} MHz")
        print(f"Bandwidth:           {self.bandwidth/1e6:.2f} MHz")
        print(f"Sweep duration:      {self.sweep_duration*1e6:.2f} us")
        print(f"Number of samples:   {self.num_samples}")
        print(f"Chirp rate:          {self.chirp_rate/1e12:.4f} THz/s")
        print(f"Range resolution:    {self.range_resolution:.4f} m")
        print(f"Max unambig. range:  {self.max_range:.2f} m")
        print(f"================================\n")

    def generate_fmcw_chirp(self):
        """Generate a complex baseband FMCW chirp signal."""
        t = np.arange(self.num_samples) / self.fs
        # Phase: 2*pi*(f_start*t + 0.5*K*t^2)
        phase = 2 * np.pi * (self.f_start * t + 0.5 * self.chirp_rate * t**2)
        chirp = np.exp(1j * phase)
        return t, chirp

    def simulate_targets(self, tx_signal, targets, snr_db=20):
        """
        Simulate received signal from multiple targets.
        
        targets: list of dicts with keys 'range' (m), 'rcs' (relative amplitude), 'velocity' (m/s, optional)
        """
        rx_signal = np.zeros_like(tx_signal, dtype=complex)
        t = np.arange(len(tx_signal)) / self.fs
        
        for tgt in targets:
            R = tgt['range']
            amp = tgt['rcs']
            vel = tgt.get('velocity', 0.0)
            
            # Round-trip time delay
            tau = 2 * R / self.c
            delay_samples = int(tau * self.fs)
            
            # Doppler frequency shift (assume carrier ~ f_start for simplicity here)
            f_carrier = self.f_start + self.bandwidth / 2
            f_doppler = 2 * vel * f_carrier / self.c
            
            # Create delayed copy of TX
            delayed = np.zeros_like(tx_signal, dtype=complex)
            if delay_samples < len(tx_signal):
                # Re-generate the TX with time shifted by tau (proper modeling)
                phase = 2 * np.pi * (
                    self.f_start * (t - tau) +
                    0.5 * self.chirp_rate * (t - tau)**2
                )
                # Apply Doppler shift
                delayed = amp * np.exp(1j * phase) * np.exp(1j * 2 * np.pi * f_doppler * t)
                # Zero-out samples before signal arrives
                delayed[:delay_samples] = 0
                rx_signal += delayed
        
        # Add complex Gaussian noise
        signal_power = np.mean(np.abs(rx_signal)**2)
        if signal_power == 0:
            signal_power = 1e-12
        noise_power = signal_power / (10**(snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal))
        )
        rx_signal += noise
        return rx_signal

    def dechirp(self, rx_signal, tx_signal):
        """
        Mix RX with conjugate of TX (dechirping / matched filter for FMCW).
        Produces beat frequency proportional to range.
        """
        if_output = rx_signal * np.conj(tx_signal)
        return if_output

    def lowpass_filter(self, x, cutoff):
        """Low-pass filter the IF signal."""
        nyq = self.fs / 2
        normalized = cutoff / nyq
        b, a = butter(4, normalized, btype='low')
        return filtfilt(b, a, x)

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
        
        # Convert beat frequency -> range
        range_axis = f_axis * self.c / (2 * self.chirp_rate)
        
        return range_axis, fft_mag, f_axis


# =====================================================
# Run Simulation
# =====================================================
def main():
    radar = FMCWRadar()
    
    # 1. Generate TX FMCW chirp
    t, tx_signal = radar.generate_fmcw_chirp()
    
    # 2. Define simulated targets (range in meters, rcs = relative amplitude)
    targets = [
        {'range': 10.0,  'rcs': 1.0,  'velocity': 0.0},
        {'range': 25.0,  'rcs': 0.7,  'velocity': 0.0},
        {'range': 50.0,  'rcs': 0.5,  'velocity': 0.0},
        {'range': 80.0,  'rcs': 0.3,  'velocity': 0.0},
    ]
    print("Simulated targets:")
    for tgt in targets:
        print(f"  Range: {tgt['range']:6.2f} m  |  RCS: {tgt['rcs']:.2f}")
    print()
    
    # 3. Simulate received signal (with noise)
    rx_signal = radar.simulate_targets(tx_signal, targets, snr_db=15)
    
    # 4. Dechirp (mix RX with conj(TX))
    if_output_raw = radar.dechirp(rx_signal, tx_signal)
    
    # 5. Low-pass filter the IF signal
    cutoff = radar.bandwidth / 2
    if_output_filt = radar.lowpass_filter(if_output_raw, cutoff)
    
    # 6. Range FFT
    range_axis, fft_mag, f_axis = radar.range_fft(if_output_filt, n_fft=4 * len(if_output_filt))
    fft_db = 20 * np.log10(fft_mag / np.max(fft_mag) + 1e-12)
    
    # 7. Peak detection
    # Only consider positive ranges
    pos_mask = range_axis >= 0
    pos_range = range_axis[pos_mask]
    pos_db = fft_db[pos_mask]
    peaks, _ = signal.find_peaks(pos_db, height=-30, distance=10)
    
    print("\nDetected peaks:")
    for p in peaks:
        print(f"  Range: {pos_range[p]:6.2f} m  |  Magnitude: {pos_db[p]:6.2f} dB")
    
    # =====================================================
    # Plotting
    # =====================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    
    # --- TX/RX time domain ---
    axs[0, 0].plot(t * 1e6, np.real(tx_signal), label='TX I', alpha=0.7)
    axs[0, 0].plot(t * 1e6, np.real(rx_signal), label='RX I', alpha=0.7)
    axs[0, 0].set_title('TX vs RX (Real part)')
    axs[0, 0].set_xlabel('Time (us)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(0, 5)  # zoom in
    
    # --- IF signal (beat) ---
    axs[0, 1].plot(t * 1e6, np.real(if_output_filt), label='IF (Real)')
    axs[0, 1].plot(t * 1e6, np.imag(if_output_filt), label='IF (Imag)', alpha=0.6)
    axs[0, 1].set_title('IF (Beat) Signal after Dechirping')
    axs[0, 1].set_xlabel('Time (us)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # --- Range Profile ---
    axs[1, 0].plot(range_axis, fft_db)
    axs[1, 0].plot(pos_range[peaks], pos_db[peaks], 'rx', markersize=10, label='Detected Peaks')
    for tgt in targets:
        axs[1, 0].axvline(tgt['range'], color='g', linestyle='--', alpha=0.5)
    axs[1, 0].set_title('Range Profile (FFT of IF signal)')
    axs[1, 0].set_xlabel('Range (m)')
    axs[1, 0].set_ylabel('Magnitude (dB)')
    axs[1, 0].set_xlim(0, radar.max_range)
    axs[1, 0].set_ylim(-80, 5)
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # --- Spectrogram of RX ---
    axs[1, 1].specgram(rx_signal, Fs=radar.fs, NFFT=256, noverlap=128, cmap='viridis')
    axs[1, 1].set_title('RX Spectrogram (Chirp visible)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    
    plt.suptitle('FMCW Radar Simulation', fontsize=14, fontweight='bold')
    plt.show()


if __name__ == '__main__':
    main()