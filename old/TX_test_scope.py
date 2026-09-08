
import time

import adi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

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


def gen_tone(fc, fs, NN):
    N = NN / 8
    fc = int(fc / (fs / N)) * (fs / N)
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 15
    q = np.sin(2 * np.pi * t * fc) * 2 ** 15
    Xn = i + 1j * q
    # Xn = np.pad(Xn, int(NN - N))
    Xn = np.pad(Xn, (0, int(NN - N)), "constant")
    return Xn


# Configure properties
print("--Setting up chip")

dev._ctx.set_timeout(3000)



dev.rx_enabled_channels = [0]
dev.tx_enabled_channels = [0]


RUNS = 100
N = int(1024*8)

dev.rx_buffer_size = N
dev.tx_cyclic_buffer = False

fs = int(dev.sample_rate)


iq1 = gen_tone(fs / 20, fs, N)

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

    dev.tx_destroy_buffer()
    dev.rx_destroy_buffer()
    dev._tx_init_channels()
    dev._rx_init_channels()

    dev.tx(iq1)

    #dev.tx_sync_start = "trigger_manual"
    input("Press Enter to continue...")
    if not ("disarm" == dev.tx_sync_start == dev.rx_sync_start):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )

    dev.rx_sync_start = "arm"
    dev.tx_sync_start = "arm"

    if not ("arm" == dev.tx_sync_start == dev.rx_sync_start):
        raise Exception(
            "Unexpected SYNC status: TX "
            + dev.tx_sync_start
            + " RX: "
            + dev.rx_sync_start
        )
    
    input("Press Enter to continue...")