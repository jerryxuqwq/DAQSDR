# Copyright (C) 2022 Analog Devices, Inc.
#
# SPDX short identifier: ADIBSD

import time

import adi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# This function is not used in the modified pulse-generation path,
# but is kept for completeness.
from sync import generate_chirp

# Connect to the DAQ2 hardware
dev = adi.DAQ2("ip:analog.local")

# --- MODIFICATION START ---
# This is the new function to generate a pulse train signal.
# It is designed to work with the ADI hardware library's requirements.

def gen_pulse(
    total_samples, pulse_width_samples, pri_samples, fs, fc, amplitude=2**15 - 1
):
    """
    Generates a pulse train signal for the ADI DAQ2.

    Parameters
    ----------
    total_samples : int
        Total number of samples in the buffer (e.g., dev.rx_buffer_size).
    pulse_width_samples : int
        The duration of a single pulse, in samples.
    pri_samples : int
        The Pulse Repetition Interval (time from start of one pulse to the next),
        in samples.
    fs : float
        The sampling frequency in Hz.
    fc : float
        The carrier frequency in Hz. A value of 0 will generate a baseband
        pulse (a simple rectangular pulse). A non-zero value will generate a
        carrier-modulated pulse.
    amplitude : float
        The peak amplitude of the signal. Should be scaled for the DAC,
        e.g., 2**15 - 1 for a 16-bit DAC.

    Returns
    -------
    signal : ndarray (complex)
        The complex baseband pulse train signal (I/Q data).
    """
    if pulse_width_samples >= pri_samples:
        raise ValueError("Pulse width (in samples) must be smaller than the PRI (in samples).")

    print(f"Generating pulse train with:")
    print(f"  - Pulse Width: {pulse_width_samples} samples ({pulse_width_samples/fs*1e6:.2f} us)")
    print(f"  - PRI: {pri_samples} samples ({pri_samples/fs*1e6:.2f} us)")
    print(f"  - Carrier Freq: {fc/1e6} MHz")
    print(f"  - Num Pulses in buffer: ~{total_samples // pri_samples}")

    # Create a sample index vector
    n = np.arange(total_samples)

    # 1. Create the rectangular pulse envelope (the "on/off" switch)
    # The modulo operator on the sample index creates the repeating pattern.
    pulse_envelope = np.where((n % pri_samples) < pulse_width_samples, 1.0, 0.0)

    # 2. Create the carrier wave
    # The time vector `t` is implicitly represented by `n / fs`.
    # Using np.exp(1j * ...) directly generates a complex signal (I+jQ).
    # If fc=0, this carrier_wave is just an array of 1s, resulting in a baseband pulse.
    carrier_wave = np.exp(1j * 2 * np.pi * fc * n / fs)

    # 3. Generate the final pulsed signal by multiplication
    radar_signal = amplitude * carrier_wave * pulse_envelope

    # Ensure the output is complex128, as expected by the adi library
    return radar_signal.astype(np.complex128)

# --- MODIFICATION END ---


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
        i = np.argmax(np.abs(cor))
        m = cor[i]
        sample_delay = len(chan0_tmp) - i - 1
        phases.append(np.angle(m) * 180 / np.pi)
        delays.append(sample_delay)
    return (np.mean(phases), np.mean(delays))


def generate_sfcw(
    f_start, f_step, num_steps, step_duration_samples, fs, amplitude=2**15
):
    samples_per_step = step_duration_samples
    total_samples = samples_per_step * num_steps
    t = np.arange(total_samples) / fs
    signal = np.zeros(total_samples, dtype=np.complex128)
    freqs = f_start + f_step * np.arange(num_steps)
    idx = 0
    for f in freqs:
        n = np.arange(samples_per_step)
        phase = 2 * np.pi * f * n / fs
        signal[idx : idx + samples_per_step] = amplitude * np.exp(1j * phase)
        idx += samples_per_step
    return t, signal, freqs


def gen_tone(fc, fs, NN):
    N_tone = NN / 8
    fc = int(fc / (fs / N_tone)) * (fs / N_tone)
    ts = 1 / float(fs)
    t = np.arange(0, N_tone * ts, ts)
    # Fixed syntax error here: * was missing
    i = np.cos(2 * np.pi * t * fc) * 2**13
    q = np.sin(2 * np.pi * t * fc) * 2**13
    Xn = i + 1j * q
    Xn = np.pad(Xn, (0, int(NN - N_tone)), "constant")
    return Xn


# Configure properties
print("--Setting up chip")

dev._ctx.set_timeout(3000)
# dev._rxadc.set_kernel_buffers_count(1)
# dev._txdac.set_kernel_buffers_count(2)

dev.rx_enabled_channels = [0]
dev.tx_enabled_channels = [0]

run_plot = True
tx_use_dma = True

RUNS = 10
N = int(1024 * 1024 / 4)

dev.rx_buffer_size = N
dev.tx_cyclic_buffer = False

fs = int(dev.sample_rate)

dev.dds_single_tone(fs / 10, 0.0, channel=0)

iq1 = gen_pulse(
    total_samples=N,
    pulse_width_samples=64, # A short pulse of 64 samples
    pri_samples=4096,       # A pulse is sent every 4096 samples
    fs=fs,
    fc=fs/20                # Carrier frequency of 1/20th the sampling rate
)

dev.tx_ddr_offload = 1
print(dev.tx_ddr_offload)

print("TX SYNC START AVAILABLE:", dev.tx_sync_start_available)
print("RX SYNC START AVAILABLE:", dev.rx_sync_start_available)

so = []
po = []

# Collect data
for r in range(RUNS):
    dev.rx_sync_start = "arm"
    dev.tx_sync_start = "arm"
    
    # Fixed syntax error in this check
    if not (dev.tx_sync_start == "arm" and dev.rx_sync_start == "arm"):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )

    dev.tx_destroy_buffer()
    dev.rx_destroy_buffer()
    dev._rx_init_channels()

    dev.tx(iq1)
    if True:
        dev.tx_sync_start = "trigger_manual"
    else:
        input("Press Enter to continue...")
        
    # Fixed syntax error in this check
    if not (dev.tx_sync_start == "disarm" and dev.rx_sync_start == "disarm"):
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

    # You can uncomment this section to re-enable phase/delay measurement
    # if tx_use_dma:
    #     (p, s) = measure_phase_and_delay(iq1[100:], x[100:])
    #     print("Run#", r, "Sample Delay", int(s), "Phase", f"{p:.2f}")
    #     so.append(s)
    #     po.append(p)

    if run_plot == True:
        plt.plot(np.real(x), label=str(r), alpha=0.7)
        plt.legend()
        plt.title("MxFE Phase Sync @ " + str(fs / 1000000) + " MSPS")
        plt.draw()
        plt.pause(0.05)
        time.sleep(0.1)

if tx_use_dma and run_plot == True:
    # Plot a small section of the generated TX pulse train to see its structure
    plt.figure()
    plt.title("Generated TX Pulse Train (First 10000 samples)")
    plt.plot(np.real(iq1[:10000]), label="TX I-component", alpha=0.8)
    plt.plot(np.imag(iq1[:10000]), label="TX Q-component", alpha=0.8, linestyle='--')
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.draw()

if run_plot == True:
    plt.show()