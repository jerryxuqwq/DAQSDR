import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, chirp,gausspulse
import pickle
from scipy.signal.windows import tukey

class FMCW_Radar:
    def __init__(self, uri="ip:analog.local",
                 num_samples=int(1024 * 1024 / 4 / 4), 
                 timeout=3000,
                 fs=1e9, 
                 f_start=-500e6,
                 f_stop=500e6,
                 tx_duration=1,
                 R_cal=0.1): 
        
        self.fs = fs
        self.f_start = f_start
        self.f_stop = f_stop
        self.duration = num_samples / fs
        self.tx_duration = tx_duration
        self.c = 3e8 
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
        print("Device Started")

    def ota(self, tx_data_complex):
        tx_data = [np.real(tx_data_complex), np.imag(tx_data_complex)]
        dev = self.dev
        dev.rx_sync_start = "arm"
        dev.tx_sync_start = "arm"
        
        if self._first_run:
            dev.tx_destroy_buffer()
            self._first_run = False
            
        dev.rx_destroy_buffer()
        dev._rx_init_channels()
        dev.tx(tx_data)
        dev.tx_sync_start = "trigger_manual"
        rx_data = dev.rx()
        return rx_data[0] * 11 + 1j * rx_data[1] * 11

    def generate_fmcw_chirp(self):
        t_whole = np.arange(0, self.duration, 1 / self.fs)
        k = (self.f_stop - self.f_start) / self.duration
        chirp_signal = np.exp(1j * 2 * np.pi * (self.f_start * t_whole + 0.5 * k * t_whole**2)) * 2**14
        return t_whole, chirp_signal

    def smooth_band(self,freqs, f_low, f_high, transition_width):
        """
        Creates a smooth window using a sigmoid-like transition.
        """
        # Smooth start
        w_low = 1 / (1 + np.exp(-(np.abs(freqs) - f_low) / (transition_width / 10)))
        # Smooth stop
        w_high = 1 - 1 / (1 + np.exp(-(np.abs(freqs) - f_high) / (transition_width / 10)))
        return w_low * w_high

    def calculate_h(self,X, Y, f_axis, band_hz, transition_ghz):
        # 1. Regularization
        reg = 1e-8 * np.max(np.abs(X)**2)
        
        # 2. Wiener-like filter (Deconvolution)
        H = Y * np.conj(X) / (np.abs(X)**2 + reg)
        
        # 3. Normalize
        H = H / np.max(np.abs(H))
        
        # 4. Smooth band window
        W = self.smooth_band(np.abs(f_axis), band_hz[0], band_hz[1], transition_ghz * 1e9)
        H = H * W
        
        return H

    def read_pkl(self, filename):
        """
        Reads time (tR) and signal (yDist) data from a pickle (.pkl) file.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Assuming the pickle file contains a tuple or list: (tR, yDist)
        # If it is a dictionary, change this to something like:
        # tR = data['time']
        # yDist = data['signal']
        return data

    def save_pkl(self, filename, a, b,c,d):
        """
        Saves time (tR) and signal (yDist) data to a pickle (.pkl) file.
        """
        data = (np.array(a), np.array(b), np.array(c), np.array(d))
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    

    def lowpass_filter(self, data, cutoff, fs, order=5):
        """
        Standard Butterworth lowpass filter.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # --- Main Signal Recovery Logic ---

    def recover_signal(self,t, dt, fs, nfft, H, bandHz, opts):
        """
        opts is assumed to be an object or dictionary containing the settings:
        opts.RecoverFile, opts.TimeTaperFrac, opts.RecoverLambda, opts.ApplyLowpass
        """
        
        # 1. Read Pickle file
        tR, yDist = self.read_pkl(opts['RecoverFile'])
        
        # 2. Interpolate if time vectors do not match lengths
        if len(tR) != len(t):
            yDist = np.interp(t, tR, yDist, left=0, right=0)
            
        # 3. Detrend (remove mean)
        yDist = yDist - np.mean(yDist)
        
        # 4. Apply Time Taper
        taper_window = tukey(len(yDist), alpha=opts['TimeTaperFrac'])
        yDist = yDist * taper_window
        
        # 5. Transform to Frequency Domain
        Z = np.fft.fftshift(np.fft.fft(yDist, n=nfft))
        
        # 6. Calculate Regularization Parameter
        lambda_reg = opts['RecoverLambda'] * np.max(np.abs(H)**2)
        
        # 7. Wiener Deconvolution / Inversion
        Xhat = Z * np.conj(H) / (np.abs(H)**2 + lambda_reg)
        
        # 9. Return to Time Domain
        xhat = np.fft.ifft(np.fft.ifftshift(Xhat), n=nfft)
            
        # 11. Generate Time Vector (nanoseconds)
        t_rec_ns = np.arange(nfft) * dt * 1e9
    
        
        return t_rec_ns, xhat

def main():
    num_samples = int(1024 * 1024 / 4 / 4)
    radar = FMCW_Radar(num_samples=num_samples, f_start=-500e6, f_stop=500e6, tx_duration=0.4)

    # 1. Generate Signal
    time, tx_signal = radar.generate_fmcw_chirp()
    
    i = np.sin(2 * np.pi * 200e6 * time)
    q = np.cos(2 * np.pi * 200e6 * time)
    
    tx_signal = (i + 1j * q) * 2**14
    # 2. Get Data
    rx_iq = radar.ota(tx_signal)
    
    rx_time = np.arange(len(rx_iq)) / radar.fs
    #radar.save_pkl("bode_recover.pkl", time,tx_signal,rx_time, rx_iq)
    print(len(tx_signal),len(rx_iq))
    # 3. Compute Transfer Function H(f) = Y(f) / X(f)
    n = len(tx_signal)  # Zero pad by 4x
    X_f = np.fft.fft(tx_signal, n=n)
    Y_f = np.fft.fft(rx_iq, n=n)

    band_hz = [-500e6, 500e6]  # Example: 1 MHz to 400 MHz

    H_f = Y_f * np.conj(X_f) / (np.abs(X_f)**2 + 1e-12) # Add epsilon to avoid div by zero
    
    # 4. Frequency Axis
    freqs = np.fft.fftfreq(n, 1 / radar.fs)
    pos_idx = freqs > 0
    f_plot = freqs[pos_idx] / 1e6 # MHz
    

    # 5. Magnitude (dB) and Phase (degrees)
    mag_db = 20 * np.log10(np.abs(H_f[pos_idx]))
    phase_deg = np.degrees(np.unwrap(np.angle(H_f[pos_idx])))


    Rtime, Rtx_signal, Rrx_time, Rrx_iq = radar.read_pkl("bode_recover.pkl")
    n = len(Rtx_signal)   # Zero pad by 4x (if you choose to pad)
    
    RX_f = np.fft.fft(Rtx_signal, n=n)
    RY_f = np.fft.fft(Rrx_iq, n=n)

    # 1. Calculate proper H(f) with dynamic regularization
    # 1e-12 is too small. We scale it relative to the max signal power.
    reg_rx = 1e-6 * np.max(np.abs(RX_f)**2)
    H_fref = RY_f * np.conj(RX_f) / (np.abs(RX_f)**2 + reg_rx)

    # 2. Calculate the Equalizer/Inverse filter with proper Wiener regularization
    inv_reg = 1e-4 * np.max(np.abs(H_fref)**2) 
    H_fcorr = np.conj(H_fref) / (np.abs(H_fref)**2 + inv_reg)

    # 3. Apply your smooth band window to KILL out-of-band noise!
    # Adjust 450e6 (450 MHz) based on your actual reliable bandwidth
    # window = radar.smooth_band(freqs, f_low=0, f_high=500e6, transition_width=50e6)
    # H_fcorr = H_fcorr * window

    # 4. Correct Y_f and recover original signal
    Xhat_f = Y_f * H_fcorr
    xhat = np.fft.ifft(Xhat_f)
    t_rec = np.arange(len(xhat)) / radar.fs

    # 5. FIX: Use Xhat_f for the frequency domain plots, NOT the time-domain xhat!
    # Added a small epsilon (1e-12) to prevent log10(0) warnings
    corr_mag_db = 20 * np.log10(np.abs(Xhat_f[pos_idx]) + 1e-12)
    corr_phase_deg = np.degrees(np.unwrap(np.angle(Xhat_f[pos_idx])))
    
    # Plot time domain signals
    fig_time, (ax_time1, ax_time2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax_time1.plot(rx_time * 1e9, np.real(rx_iq), label='RX Signal (Real)')
    ax_time1.plot(rx_time * 1e9, np.imag(rx_iq), label='RX Signal (Imaginary)')
    ax_time1.set_ylabel('Amplitude')
    ax_time1.set_title('Time Domain Signal - Before Recovery')
    ax_time1.grid(True, linestyle='--')
    
    ax_time2.plot(t_rec * 1e9, np.real(xhat), label='Recovered Signal (Real)')
    ax_time2.plot(t_rec * 1e9, np.imag(xhat), label='Recovered Signal (Imaginary)')
    ax_time2.set_ylabel('Amplitude')
    ax_time2.set_title('Time Domain Signal - After Recovery')
    ax_time2.grid(True, linestyle='--')

    
    plt.tight_layout()
    plt.show()
    # 6. Plotting
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogx(f_plot, mag_db)
    ax1.semilogx(f_plot, corr_mag_db, label='Recovered Signal', linestyle='--')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('System Transfer Function (Bode Plot)')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend()

    ax2.semilogx(f_plot, phase_deg)
    ax2.semilogx(f_plot, corr_phase_deg, label='Recovered Signal', linestyle='--')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()