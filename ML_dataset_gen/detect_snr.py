import pickle
import numpy as np

def load_radioml_dataset(filename='RML2014.04c_dict.dat'):
    """
    Load RadioML dataset (.dat file - pickled dictionary).
    Keys are tuples of (modulation_type, SNR_in_dB)
    Values are numpy arrays of shape (n_samples, 2, 128) 
        - 2 channels: I and Q
        - 128 samples per example
    """
    with open(filename, 'rb') as f:
        # RadioML datasets are typically pickled with latin1 encoding (Python 2 -> 3)
        data = pickle.load(f, encoding='latin1')
    return data


def compute_snr_from_iq(iq_signal, noise_estimate=None):
    """
    Compute SNR from an I/Q signal.
    
    iq_signal: numpy array of shape (2, N) - I and Q channels
               or (N,) complex array
    
    Returns SNR in dB.
    """
    # Convert to complex if needed
    if iq_signal.ndim == 2 and iq_signal.shape[0] == 2:
        complex_signal = iq_signal[0] + 1j * iq_signal[1]
    else:
        complex_signal = iq_signal
    
    # Signal power (mean squared magnitude)
    signal_power = np.mean(np.abs(complex_signal) ** 2)
    
    if noise_estimate is None:
        # Estimate noise from high-frequency components via FFT
        # Assume signal occupies center, noise is spread
        spectrum = np.fft.fftshift(np.fft.fft(complex_signal))
        mag = np.abs(spectrum) ** 2
        # Use lowest 20% of bins as noise floor estimate
        sorted_mag = np.sort(mag)
        noise_power = np.mean(sorted_mag[:len(sorted_mag) // 5])
        signal_power_clean = signal_power - noise_power
        if signal_power_clean <= 0:
            signal_power_clean = signal_power
    else:
        noise_power = noise_estimate
        signal_power_clean = signal_power
    
    snr_db = 10 * np.log10(signal_power_clean / noise_power)
    return snr_db


def main():
    # Load the dataset
    print("Loading RML2014.04c_dict.dat ...")
    data = load_radioml_dataset('RML2014.04c_dict.dat')
    
    # Inspect dataset structure
    keys = list(data.keys())
    print(f"Number of (modulation, SNR) keys: {len(keys)}")
    print(f"First 5 keys: {keys[:5]}")
    
    # Get unique modulations and SNRs
    mods = sorted(set(k[0] for k in keys))
    snrs = sorted(set(k[1] for k in keys))
    print(f"Modulations ({len(mods)}): {mods}")
    print(f"SNRs (dB) ({len(snrs)}): {snrs}")
    
    # Get the FIRST test sample
    first_key = keys[0]
    print(f"\n--- First test ---")
    print(f"Key (modulation, SNR_label): {first_key}")
    
    samples = data[first_key]
    print(f"Shape of samples for this key: {samples.shape}")
    
    # The first individual signal example
    first_signal = samples[0]   # shape (2, 128)
    print(f"Shape of first signal: {first_signal.shape}")
    
    # The dataset already labels SNR in the key
    labeled_snr = first_key[1]
    print(f"\nLabeled SNR (from dataset key): {labeled_snr} dB")
    
    # Empirically compute SNR from the IQ samples
    estimated_snr = compute_snr_from_iq(first_signal)
    print(f"Estimated SNR from IQ data: {estimated_snr:.2f} dB")
    
    # Optional: plot
    try:
        import matplotlib.pyplot as plt
        I = first_signal[0]
        Q = first_signal[1]
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(I, label='I')
        axs[0].plot(Q, label='Q')
        axs[0].set_title(f"First test: mod={first_key[0]}, labeled SNR={labeled_snr} dB")
        axs[0].legend(); axs[0].grid(True)
        
        complex_sig = I + 1j * Q
        spectrum = np.fft.fftshift(np.fft.fft(complex_sig))
        axs[1].plot(20 * np.log10(np.abs(spectrum) + 1e-12))
        axs[1].set_title("Spectrum (dB)")
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == '__main__':
    main()