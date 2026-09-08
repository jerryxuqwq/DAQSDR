import numpy as np
import matplotlib.pyplot as plt

class WaveformGenerator:
    """
    A class to generate radar waveforms using sample indices and sample counts.
    """

    def __init__(self, num_samples: int, fs: float, amplitude: float = 1.0):
        """
        Initializes the WaveformGenerator.

        Args:
            num_samples (int): Total number of samples in the generated buffer.
            fs (float): Sampling frequency in Hertz.
            amplitude (float, optional): Peak amplitude of the waveform.
        """
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("fs (sampling frequency) must be a positive number.")

        self.num_samples = num_samples
        self.fs = fs
        self.amplitude = amplitude
        self.n = np.arange(self.num_samples)

    def generate_pulse(
        self,
        fc_cycles_per_sample: float,
        pulse_width_samples: int,
        pri_samples: int,
        start_sample: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a pulse train based on sample counts.

        Args:
            fc_cycles_per_sample (float): Carrier frequency in cycles per sample.
            pulse_width_samples (int): Pulse width in samples.
            pri_samples (int): Pulse repetition interval in samples.
            start_sample (int): Sample index where the first pulse begins.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sample indices and complex I/Q signal.
        """
        if not isinstance(pulse_width_samples, int) or pulse_width_samples <= 0:
            raise ValueError("pulse_width_samples must be a positive integer.")
        if not isinstance(pri_samples, int) or pri_samples <= 0:
            raise ValueError("pri_samples must be a positive integer.")
        if not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("start_sample must be a non-negative integer.")
        if pulse_width_samples >= pri_samples:
            raise ValueError("Pulse width samples must be smaller than pri_samples.")
        if start_sample >= self.num_samples:
            raise ValueError("start_sample must be within the waveform length.")

        active = (self.n >= start_sample) & (((self.n - start_sample) % pri_samples) < pulse_width_samples)
        pulse_envelope = np.where(active, 1.0, 0.0)
        carrier_wave = np.exp(1j * 2 * np.pi * fc_cycles_per_sample * self.n)
        signal = self.amplitude * pulse_envelope * carrier_wave
        return self.n, signal

    def generate_cw(
        self,
        fc_cycles_per_sample: float,
        start_sample: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a continuous wave using sample-rate-normalized frequency.

        Args:
            fc_cycles_per_sample (float): Carrier frequency in cycles per sample.
            start_sample (int): Sample index at which the CW starts.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sample indices and complex I/Q signal.
        """
        if not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("start_sample must be a non-negative integer.")
        if start_sample >= self.num_samples:
            raise ValueError("start_sample must be within the waveform length.")

        waveform = np.exp(1j * 2 * np.pi * fc_cycles_per_sample * self.n)
        waveform[:start_sample] = 0.0
        signal = self.amplitude * waveform
        return self.n, signal

    def generate_fmcw(
        self,
        f_start_cycles_per_sample: float,
        sweep_duration_samples: int,
        bandwidth_cycles_per_sample: float,
        start_sample: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate an FMCW chirp using sample counts for duration.

        Args:
            f_start_cycles_per_sample (float): Start frequency in cycles per sample.
            sweep_duration_samples (int): Sweep duration in samples.
            bandwidth_cycles_per_sample (float): Total sweep bandwidth in cycles per sample.
            start_sample (int): Sample index at which the chirp starts.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sample indices and complex I/Q signal.
        """
        if not isinstance(sweep_duration_samples, int) or sweep_duration_samples <= 0:
            raise ValueError("sweep_duration_samples must be a positive integer.")
        if not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("start_sample must be a non-negative integer.")
        if start_sample >= self.num_samples:
            raise ValueError("start_sample must be within the waveform length.")

        sweep_index = np.maximum(self.n - start_sample, 0)
        normalized_index = sweep_index % sweep_duration_samples
        chirp_rate = bandwidth_cycles_per_sample / sweep_duration_samples
        phase = 2 * np.pi * (
            f_start_cycles_per_sample * normalized_index
            + 0.5 * chirp_rate * normalized_index**2
        )
        signal = self.amplitude * np.exp(1j * phase)
        signal[self.n < start_sample] = 0.0
        return self.n, signal


# --- USAGE EXAMPLE AND VISUALIZATION ---
if __name__ == '__main__':
    SAMPLING_FREQ = 1000e6  # 1000 MHz
    NUM_SAMPLES = int(1e6)  # total buffer length in samples
    AMPLITUDE = 1.0

    generator = WaveformGenerator(num_samples=NUM_SAMPLES, fs=SAMPLING_FREQ, amplitude=AMPLITUDE)

    pulse_t, pulse_signal = generator.generate_pulse(
        fc_cycles_per_sample=5e6 / SAMPLING_FREQ,
        pulse_width_samples=100,
        pri_samples=1000,
        start_sample=1,
    )

    cw_t, cw_signal = generator.generate_cw(
        fc_cycles_per_sample=2e6 / SAMPLING_FREQ,
        start_sample=0,
    )

    fmcw_t, fmcw_signal = generator.generate_fmcw(
        f_start_cycles_per_sample=2e6 / SAMPLING_FREQ,
        sweep_duration_samples=1000,
        bandwidth_cycles_per_sample=5e6 / SAMPLING_FREQ,
        start_sample=0,
    )

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Radar Waveform Generation (Sample-Based)', fontsize=16)

    axes[0, 0].set_title('Pulse Train (Sample Domain)')
    axes[0, 0].plot(pulse_t, np.real(pulse_signal), label='I (Real)')
    axes[0, 0].plot(pulse_t, np.imag(pulse_signal), label='Q (Imag)', alpha=0.7)
    axes[0, 0].set_xlabel('Sample index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    fft_pulse = np.fft.fftshift(np.fft.fft(pulse_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(NUM_SAMPLES, 1 / SAMPLING_FREQ))
    axes[0, 1].set_title('Pulse Train (Frequency Domain)')
    axes[0, 1].plot(freqs / 1e6, 20 * np.log10(np.abs(fft_pulse)))
    axes[0, 1].set_xlabel('Frequency (MHz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].grid(True)

    axes[1, 0].set_title('Continuous Wave (Sample Domain)')
    axes[1, 0].plot(cw_t[:500], np.real(cw_signal[:500]), label='I (Real)')
    axes[1, 0].plot(cw_t[:500], np.imag(cw_signal[:500]), label='Q (Imag)', alpha=0.7)
    axes[1, 0].set_xlabel('Sample index')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    fft_cw = np.fft.fftshift(np.fft.fft(cw_signal))
    axes[1, 1].set_title('Continuous Wave (Frequency Domain)')
    axes[1, 1].plot(freqs / 1e6, 20 * np.log10(np.abs(fft_cw)))
    axes[1, 1].set_xlabel('Frequency (MHz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].grid(True)

    axes[2, 0].set_title('FMCW Chirp (Sample Domain)')
    axes[2, 0].plot(fmcw_t[:1000], np.real(fmcw_signal[:1000]), label='I (Real)')
    axes[2, 0].plot(fmcw_t[:1000], np.imag(fmcw_signal[:1000]), label='Q (Imag)', alpha=0.7)
    axes[2, 0].set_xlabel('Sample index')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].set_title('FMCW Chirp (Spectrogram)')
    axes[2, 1].specgram(fmcw_signal, Fs=SAMPLING_FREQ, NFFT=256, noverlap=128)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Frequency (Hz)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
