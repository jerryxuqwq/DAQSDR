# Copyright (C) 2022 Analog Devices, Inc.
#
# SPDX short identifier: ADIBSD

import time

import adi
import numpy as np
import seaborn as sns
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from sync import generate_chirp
plt.style.use(['ieee'])

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titlesize': 8,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.size': 6
})

dev = adi.DAQ2("ip:analog.local")

def measure_phase_and_delay(chan0, chan1, window=None):
    assert len(chan0) == len(chan1)
    if window == None:
        window = len(chan0)
    phases = []
    delays = []
    indx = 0
    sections = len(chan0) // window
    for sec in range(sections):
        chan0_tmp = chan0[indx : indx + window]
        chan1_tmp = chan1[indx : indx + window]
        indx = indx + window + 1
        cor = np.correlate(chan0_tmp, chan1_tmp, "full")
        # plt.plot(np.real(cor))
        # plt.plot(np.imag(cor))
        # plt.plot(np.abs(cor))
        # plt.show()
        i = np.argmax(np.abs(cor))
        m = cor[i]
        sample_delay = len(chan0_tmp) - i - 1
        phases.append(np.angle(m) * 180 / np.pi)
        delays.append(sample_delay)
    return (np.mean(phases), np.mean(delays))

def generate_sfcw(
    f_start,
    f_step,
    num_steps,
    step_duration_samples,
    fs,
    amplitude=2 ** 15
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
    step_duration_samples : int
        Duration of each frequency step (number of samples)
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

    samples_per_step = step_duration_samples
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


def gen_tone(fc, fs, NN):
    N = NN/8 
    fc = int(fc / (fs / N)) * (fs / N)
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 13
    q = np.sin(2 * np.pi * t * fc) * 2 ** 13
    Xn = i + 1j * q
    #Xn = np.pad(Xn, int(NN - N))
    Xn = np.pad(Xn, (0, int(NN - N)), "constant")

    return Xn

# def gen_tone(fc, fs, NN):
#     ts = 1 / float(fs)
#     t = np.arange(0, NN * ts, ts)
#     i = np.cos(2 * np.pi * t * fc)
#     q = np.sin(2 * np.pi * t * fc)
#     return i + 1j * q
# Configure properties
print("--Setting up chip")

dev._ctx.set_timeout(3000)
dev._rxadc.set_kernel_buffers_count(1)
dev._txdac.set_kernel_buffers_count(2)



dev.rx_enabled_channels = [0]
dev.tx_enabled_channels = [0]

run_plot = True
tx_use_dma = True

RUNS = 3
N = int(1024*1024/4/4)

dev.rx_buffer_size = N
dev.tx_cyclic_buffer = False

fs = int(dev.sample_rate)

dev.dds_single_tone(fs / 10, 0.0, channel=0)

if tx_use_dma:
    iq1 = gen_tone(fs / 1000, fs, N)
    #tx_i, tx_q = generate_chirp(fs, N, fs / 10, fs, int(1024*8))
    #iq1 = tx_i + 1j * tx_q
    #iq1 = gen_pulse(N, pulse_width=8, fs=fs, fc=0)
else:
    # Set single DDS tone for TX on one transmitter
    dev.dds_single_tone(fs / 10, 0.5, channel=0)
    dev.dds_single_tone(fs / 10, 0.5, channel=1)
dev.tx_ddr_offload = 1
print(dev.tx_ddr_offload)

print("TX SYNC START AVAILABLE:", dev.tx_sync_start_available)
print("RX SYNC START AVAILABLE:", dev.rx_sync_start_available)

# dev._rxadc.reg_write(0x4a3, 25)

so = []
po = []

# Collect data
for r in range(RUNS):
    dev.rx_sync_start = "arm"
    dev.tx_sync_start = "arm"

    if not ("arm" == dev.tx_sync_start == dev.rx_sync_start):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )
    #if r==0:
    dev.tx_destroy_buffer()
    dev.rx_destroy_buffer()
    #dev._tx_init_channels()
    dev._rx_init_channels()
    

    dev.tx(iq1)
    if True:
        dev.tx_sync_start = "trigger_manual"
    else:
        input("Press Enter to continue...")
    if not ("disarm" == dev.tx_sync_start == dev.rx_sync_start):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )

    try:
        x = dev.rx()
        print("DMA transfer complete.")
    except Exception as e:
        print("Run#", r, " ----------------------------- FAILED:", e)
        continue

    if run_plot == True:
        # plt.xlim(350, 450)
        
        plt.plot(np.real(x)*10, label=str(r), alpha=0.7)
        #plt.legend(fontsize=10)
        #plt.title(" Phase Sync @ " + str(fs / 1000000) + " MSPS")
        plt.minorticks_on()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Samples',fontweight='bold', labelpad=10, fontsize=18)
        plt.ylabel('Amplitude',fontweight='bold',labelpad=10, fontsize=18)
        #plt.draw()
        #plt.pause(0.05)
        time.sleep(0.1)


if run_plot == True:

    plt.tight_layout()
    plt.savefig('delay.png', dpi=300, bbox_inches='tight')