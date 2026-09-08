import numpy as np
C0 = 299_792_458.0  # m/s
def matched_filter(rx_iq, tx_iq_ref, SAMPLE_RATE, NUM_SAMPLES): 
# --- Matched Filter: Mix RX with TX (complex multiplication) ---

    # Pad tx_iq_ref to match rx_iq length if needed
    tx_iq_ref_padded = np.pad(tx_iq_ref[::-1], (0, max(0, len(rx_iq) - len(tx_iq_ref))), mode='constant')
    if_output = rx_iq * tx_iq_ref_padded
    #if_output = if_output[0:NUM_SAMPLES]  # Ensure we only take the expected number of samples for plotting
    if_average = np.average(np.abs(if_output))
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(tx_iq_ref)
    # plt.title('TX IQ')
    # plt.xlabel('Sample')
    # plt.ylabel('Magnitude')
    # plt.subplot(1, 2, 2)
    # plt.plot(tx_iq_ref_padded)
    # plt.title('TX Padded')
    # plt.xlabel('Sample')
    # plt.ylabel('Magnitude')
    # plt.tight_layout()
    # plt.show()
    return if_output, if_average

def range_matched_filter_1d(beat, fs, slope, nfft=None, window=True):
    """
    Matched filter in fast-time for FMCW after dechirp is essentially an FFT
    (beat frequency -> range). This returns range spectrum and range axis.

    beat: shape (..., Ns) where last axis is fast-time
    """
    Ns = beat.shape[-1]
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(Ns)))

    w = np.hanning(Ns) if window else np.ones(Ns)
    X = np.fft.fft(beat * w, n=nfft, axis=-1)
    X = np.fft.fftshift(X, axes=-1)

    # Frequency axis (shifted)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    # Beat freq fb maps to range: R = c * fb / (2 * slope)
    R = C0 * f / (2 * slope)
    return X, R