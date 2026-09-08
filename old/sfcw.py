import numpy as np

def generate_sfcw(
    f_start,
    f_step,
    num_steps,
    step_duration,
    fs,
    amplitude=1.0
):
    """
    Generate a Stepped-Frequency Continuous Wave (SFCW) radar waveform.

    Parameters
    ----------
    f_start : float
        Starting frequency (Hz)
    f_step : float
        Frequency step size (Hz)
    num_steps : int
        Number of frequency steps
    step_duration : float
        Duration of each frequency step (seconds)
    fs : float
        Sampling frequency (Hz)
    amplitude : float
        Signal amplitude

    Returns
    -------
    t : ndarray
        Time vector
    signal : ndarray (complex)
        Complex baseband SFCW signal
    freqs : ndarray
        Frequency used at each step
    """

    samples_per_step = int(step_duration * fs)
    total_samples = samples_per_step * num_steps

    t = np.arange(total_samples) / fs
    signal = np.zeros(total_samples, dtype=np.complex128)

    freqs = f_start + f_step * np.arange(num_steps)

    idx = 0
    for f in freqs:
        n = np.arange(samples_per_step)
        phase = 2 * np.pi * f * n / fs
        signal[idx:idx + samples_per_step] = amplitude * np.exp(1j * phase)
        idx += samples_per_step

    return t, signal, freqs