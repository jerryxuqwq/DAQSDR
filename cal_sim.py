import numpy as np
import matplotlib.pyplot as plt

def calculate_a_calibrated(a_IF, a_IF_c, t, f_IF_c=None, W=None, R_cal=None, c=299792458.0):
    """
    Calculates the calibrated signal a_calibrated(t) according to:
    a_calibrated(t) = [a_IF(t) / a_IF,c(t)] * exp(2 * pi * j * f_IF,c * t)
                    = [a_IF(t) / a_IF,c(t)] * exp((4 * pi * j * W * R_cal * t) / c)
    """

    # Compute phase factor using either f_IF_c directly or via W and R_cal
    if f_IF_c is not None:
        exponent = 2 * np.pi * 1j * f_IF_c * t
    elif W is not None and R_cal is not None:
        exponent = (4 * np.pi * 1j * W * R_cal * t) / c
    else:
        raise ValueError("You must provide either 'f_IF_c' or both 'W' and 'R_cal'.")

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(
            a_IF,
            a_IF_c,
            out=np.zeros_like(a_IF, dtype=np.complex128),
            where=a_IF_c != 0,
        )

    return ratio * np.exp(exponent)


def run_fmcw_simulation():
    # ---------------------------------------------------------
    # 1. Radar & Simulation Parameters
    # ---------------------------------------------------------
    c = 299792458.0      # Speed of light (m/s)
    B = 1e9              # Bandwidth: 1 GHz
    T_c = 100e-6         # Chirp duration: 100 us
    W = B / T_c          # Chirp slope: 1e13 Hz/s
    fs = 100e6           # Sampling rate: 100 MHz
    
    # Time vector
    t = np.arange(0, T_c, 1/fs)
    N = len(t)

    # Target parameters
    R_target = 12.5      # Distance to target (m)
    R_cal = 3.0          # Distance to calibration target (m)

    # Intermediate Frequencies (f_IF = 2 * W * R / c)
    f_IF_target = (2 * W * R_target) / c
    f_IF_cal = (2 * W * R_cal) / c

    # ---------------------------------------------------------
    # 2. Simulate Signal Distortion & Noise
    # ---------------------------------------------------------
    # Non-linear chirp error (quadratic phase distortion + ripple)
    # This phase distortion is common to both measurements.
    phi_error = 80 * (t / T_c)**2 + 8 * np.sin(2 * np.pi * 4 * t / T_c)

    # Add Gaussian noise
    snr_db = 25
    noise_scale = 10 ** (-snr_db / 20) / np.sqrt(2)
    noise_target = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_scale
    noise_cal = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_scale

    # Generate Beat/IF Signals (Complex analytic representation)
    a_IF = np.exp(1j * (2 * np.pi * f_IF_target * t + phi_error)) + noise_target
    a_IF_c = np.exp(1j * (2 * np.pi * f_IF_cal * t + phi_error)) + noise_cal

    # ---------------------------------------------------------
    # 3. Apply Calibration Function
    # ---------------------------------------------------------
    a_calibrated = calculate_a_calibrated(
        a_IF=a_IF, 
        a_IF_c=a_IF_c, 
        t=t, 
        W=W, 
        R_cal=R_cal, 
        c=c
    )

    # ---------------------------------------------------------
    # 4. Range FFT Processing
    # ---------------------------------------------------------
    window = np.hanning(N)
    n_fft = 4 * N  # Zero-padding for fine range bin resolution

    # Compute FFT
    fft_raw = np.fft.fft(a_IF * window, n=n_fft)
    fft_cal = np.fft.fft(a_calibrated * window, n=n_fft)

    # Frequency and Range axis calculation
    freq_axis = np.fft.fftfreq(n_fft, d=1/fs)
    range_axis = (freq_axis * c) / (2 * W)

    # Keep positive frequencies (positive range)
    pos_mask = range_axis >= 0
    r_axis = range_axis[pos_mask]
    
    # Calculate Magnitude in dB (normalized)
    mag_raw_db = 20 * np.log10(np.abs(fft_raw[pos_mask]) + 1e-12)
    mag_cal_db = 20 * np.log10(np.abs(fft_cal[pos_mask]) + 1e-12)
    mag_raw_db -= np.max(mag_raw_db)
    mag_cal_db -= np.max(mag_cal_db)

    # ---------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))

    # Time Domain Plot (Real part comparison)
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e6, np.real(a_IF), label="Uncalibrated IF (Target @ 12.5m)", alpha=0.7)
    plt.plot(t * 1e6, np.real(a_calibrated), label="Calibrated IF", alpha=0.8, color='green')
    plt.title("Time-Domain Signals")
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 15)  # Zoomed view
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right")

    # Range Profile Plot
    plt.subplot(2, 1, 2)
    plt.plot(r_axis, mag_raw_db, label="Raw / Uncalibrated Range Profile", color="red", alpha=0.7)
    plt.plot(r_axis, mag_cal_db, label="Calibrated Range Profile", color="blue", linewidth=1.8)
    plt.axvline(R_target, color='black', linestyle=':', label=f"True Target ({R_target}m)")
    
    plt.title("FMCW Range Profile (FFT Spectrum)")
    plt.xlabel("Range (meters)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.xlim(0, 25)
    plt.ylim(-40, 5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fmcw_simulation()