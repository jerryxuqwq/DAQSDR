import adi
import time
from scipy import signal
import WaveformGenerator
import matplotlib.pyplot as plt
import numpy as np

# from MatchFilter import matched_filter
# from MatchFilter import range_matched_filter_1d

# 1. Define an array of sample sizes to test the delay against
sample_sizes = np.arange(1024, 1024*1024/4, 1024*5, dtype=int)  # Adjust step as needed for more granularity

# Arrays to store plotting data
valid_sizes = []
mean_delays = []
std_delays = []

NUM_TRIALS = 10
AMPLITUDE = 1.0

dev = adi.DAQ2("ip:analog.local")

dev._ctx.set_timeout(3000)
dev._rxadc.set_kernel_buffers_count(1)
dev._txdac.set_kernel_buffers_count(2)
dev.rx_enabled_channels = [0, 1]
dev.tx_enabled_channels = [0, 1]
dev.tx_cyclic_buffer = False
fs = int(dev.sample_rate)
dev.dds_single_tone(fs / 10, 0.0, channel=0)
print("Device Started")

# --- SPECTROGRAM SETUP (Keeping your original setup) ---
WIN_LEN = 1024
win = signal.windows.hann(WIN_LEN)
HOP = 16
SFT = signal.ShortTimeFFT(
    win,
    hop=HOP,
    fs=fs,
    fft_mode="centered",
    scale_to="psd",
)

plt.ion()

# 2. Iterate through the different sample sizes
for run_index, num_samples in enumerate(sample_sizes):
    print(f"\n--- Testing Run {run_index+1}/{len(sample_sizes)} | Samples: {num_samples} ---")
    
    # Must destroy buffers before changing rx_buffer_size
    try:
        dev.tx_destroy_buffer()
        dev.rx_destroy_buffer()
    except Exception:
        pass # Ignore if buffers don't exist yet
        
    dev.rx_buffer_size = int(num_samples)
    dev._rx_init_channels()

    # Generate waveform specifically for this sample size (only needs to be done once per size)
    generator = WaveformGenerator.WaveformGenerator(num_samples=int(num_samples), fs=fs, amplitude= 2 ** 13)
    
    f_start_cycles_per_sample = 0.1  
    sweep_duration_samples = int(num_samples)  
    bandwidth_cycles_per_sample = 0.4  

    FMCW = generator.generate_fmcw(
        f_start_cycles_per_sample=f_start_cycles_per_sample,  
        sweep_duration_samples=sweep_duration_samples,  
        bandwidth_cycles_per_sample=bandwidth_cycles_per_sample 
    )[1]

    real = np.real(FMCW)
    imag = np.imag(FMCW)

    tx_iq = [real, imag]
    
    trial_delays = []
    
    # 3. Run the transaction multiple times
    for trial in range(NUM_TRIALS):
        start = time.perf_counter()
        dev.rx_sync_start = "arm"
        dev.tx_sync_start = "arm"
        dev.tx(tx_iq)
        dev.tx_sync_start = "trigger_manual"
        x = dev.rx()
        end = time.perf_counter()
        
        elapsed_time = end - start
        trial_delays.append(elapsed_time)
        print(f"  Trial {trial+1} completed in {elapsed_time:.4f} seconds")

    # Calculate average and std dev if we had successful runs
    if trial_delays:
        avg_delay = np.mean(trial_delays)
        std_delay = np.std(trial_delays)
        
        valid_sizes.append(num_samples)
        mean_delays.append(avg_delay)
        std_delays.append(std_delay)
        print(f"> Average Delay: {avg_delay:.4f}s ± {std_delay:.4f}s")
    else:
        print(f"> All trials failed for sample size {num_samples}.")

# 4. Plot the results with error bars
plt.ioff() # Turn off interactive mode for the final plot
plt.figure(figsize=(10, 6))

sample_time = np.array(valid_sizes) / fs
# Use errorbar instead of standard plot
plt.errorbar(
    sample_time, 
    mean_delays, 
    yerr=std_delays, 
    marker='o', 
    linestyle='-', 
    color='b', 
    ecolor='r',       # Color of the error bars
    capsize=4,        # Adds horizontal caps to the error bars
    label='Mean Delay ± 1 Std Dev'
)

plt.title("Hardware TX/RX Transaction Delay vs. Buffer Time (s)")
plt.xlabel("Buffer Time (s)")
plt.ylabel("Delay (Seconds)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()