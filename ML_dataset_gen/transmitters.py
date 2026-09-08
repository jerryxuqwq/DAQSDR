#!/usr/bin/env python3
import time
import math
from scipy.signal import get_window
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes

sps = 8
ebw = 0.35

# ---------------------------------------------------------
# Constellation Generation Helpers (Replacing gr-mapper)
# ---------------------------------------------------------

def get_psk_points(M):
    # Standard M-PSK constellation generation.
    # QPSK is often rotated by pi/4 in practice, so we apply that offset.
    offset = math.pi / 4 if M == 4 else 0.0
    return [math.cos(2 * math.pi * i / M + offset) + 1j * math.sin(2 * math.pi * i / M + offset) for i in range(M)]

def get_pam_points(M):
    # Standard M-PAM constellation generation centered at 0
    return [(2 * i - M + 1) + 0j for i in range(M)]

def get_qam_points(M):
    # Standard M-QAM constellation generation
    n = int(math.sqrt(M))
    points = []
    for j in range(n):
        for i in range(n):
            x = (2 * i - n + 1)
            y = (2 * j - n + 1)
            points.append(x + 1j * y)
    return points

def get_mapped_points(points, symvals):
    # Reorder the generated points using the provided symbol value mapping
    mapped = [points[s] for s in symvals]
    
    # Normalize the average energy to 1.0
    avg_energy = sum(abs(p)**2 for p in mapped) / len(mapped)
    scale = 1.0 / math.sqrt(avg_energy)
    return [p * scale for p in mapped]

# ---------------------------------------------------------
# Transmitter Definitions
# ---------------------------------------------------------

class transmitter_mapper(gr.hier_block2):
    def __init__(self, points, txname, samples_per_symbol=2, excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        # Use GNU Radio's built-in chunks_to_symbols_bc instead of gr-mapper
        self.mod = digital.chunks_to_symbols_bc(points)
        
        # Pulse shaping filter
        nfilts = 32
        ntaps = nfilts * 11 * int(samples_per_symbol)    # Make nfilts filters of ntaps each
        
        rrc_taps = filter.firdes.root_raised_cosine(
            nfilts,          # gain
            nfilts,          # sampling rate based on 32 filters in resampler
            1.0,             # symbol rate
            excess_bw,       # excess bandwidth (roll-off factor)
            ntaps
        )
        
        self.rrc_filter = filter.pfb_arb_resampler_ccf(samples_per_symbol, rrc_taps)
        self.connect(self, self.mod, self.rrc_filter, self)
        # self.rate = const.bits_per_symbol()

class transmitter_bpsk(transmitter_mapper):
    modname = "BPSK"
    def __init__(self):
        points = get_mapped_points(get_psk_points(2), [0, 1])
        transmitter_mapper.__init__(self, points, "transmitter_bpsk", sps, ebw)

class transmitter_qpsk(transmitter_mapper):
    modname = "QPSK"
    def __init__(self):
        points = get_mapped_points(get_psk_points(4), [0, 1, 3, 2])
        transmitter_mapper.__init__(self, points, "transmitter_qpsk", sps, ebw)

class transmitter_8psk(transmitter_mapper):
    modname = "8PSK"
    def __init__(self):
        points = get_mapped_points(get_psk_points(8), [0, 1, 3, 2, 7, 6, 4, 5])
        transmitter_mapper.__init__(self, points, "transmitter_8psk", sps, ebw)

class transmitter_pam4(transmitter_mapper):
    modname = "PAM4"
    def __init__(self):
        points = get_mapped_points(get_pam_points(4), [0, 1, 3, 2])
        transmitter_mapper.__init__(self, points, "transmitter_pam4", sps, ebw)

class transmitter_qam16(transmitter_mapper):
    modname = "QAM16"
    def __init__(self):
        symvals = [2, 6, 14, 10, 3, 7, 15, 11, 1, 5, 13, 9, 0, 4, 12, 8]
        points = get_mapped_points(get_qam_points(16), symvals)
        transmitter_mapper.__init__(self, points, "transmitter_qam16", sps, ebw)

class transmitter_qam64(transmitter_mapper):
    modname = "QAM64"
    def __init__(self):
        symvals = [0, 32, 8, 40, 3, 35, 11, 43,
                   48, 16, 56, 24, 51, 19, 59, 27,
                   12, 44, 4, 36, 15, 47, 7, 39,
                   60, 28, 52, 20, 63, 31, 55, 23,
                   2, 34, 10, 42, 1, 33, 9, 41,
                   50, 18, 58, 26, 49, 17, 57, 25,
                   14, 46, 6, 38, 13, 45, 5, 37,
                   62, 30, 54, 22, 61, 29, 53, 21]
        points = get_mapped_points(get_qam_points(64), symvals)
        transmitter_mapper.__init__(self, points, "transmitter_qam64", sps, ebw)

class transmitter_gfsk(gr.hier_block2):
    modname = "GFSK"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_gfsk",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        self.repack = blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST)
        self.mod = digital.gfsk_mod(sps, sensitivity=0.1, bt=ebw)
        self.connect(self, self.repack, self.mod, self)

class transmitter_cpfsk(gr.hier_block2):
    modname = "CPFSK"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_cpfsk",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        self.mod = analog.cpfsk_bc(0.5, 1.0, sps)
        self.connect(self, self.mod, self)

class transmitter_fm(gr.hier_block2):
    modname = "WBFM"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_fm",
            gr.io_signature(1, 1, gr.sizeof_float),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        self.mod = analog.wfm_tx(audio_rate=44100.0, quad_rate=220.5e3)
        self.connect(self, self.mod, self)
        self.rate = 200e3 / 44.1e3

class transmitter_am(gr.hier_block2):
    modname = "AM-DSB"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_am",
            gr.io_signature(1, 1, gr.sizeof_float),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        self.rate = 44.1e3 / 200e3
        # UPDATED: Use mmse_resampler_ff for GNU Radio 3.10+
        self.interp = filter.mmse_resampler_ff(0.0, self.rate)
        self.cnv = blocks.float_to_complex()
        self.mul = blocks.multiply_const_cc(1.0)
        self.add = blocks.add_const_cc(1.0)
        self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_cc()
        
        self.connect(self, self.interp, self.cnv, self.mul, self.add, self.mod, self)
        self.connect(self.src, (self.mod, 1))

class transmitter_amssb(gr.hier_block2):
    modname = "AM-SSB"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_amssb",
            gr.io_signature(1, 1, gr.sizeof_float),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        self.rate = 44.1e3 / 200e3
        # UPDATED: Use mmse_resampler_ff for GNU Radio 3.10+
        self.interp = filter.mmse_resampler_ff(0.0, self.rate)
        self.mul = blocks.multiply_const_ff(1.0)
        self.add = blocks.add_const_ff(1.0)
        self.src = analog.sig_source_f(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_ff()
        self.filt = filter.hilbert_fc(401)
        
        self.connect(self, self.interp, self.mul, self.add, self.mod, self.filt, self)
        self.connect(self.src, (self.mod, 1))

transmitters = {
    "discrete": [transmitter_bpsk, transmitter_qpsk, transmitter_8psk, transmitter_pam4, transmitter_qam16, transmitter_qam64, transmitter_gfsk, transmitter_cpfsk],
    "continuous": [transmitter_fm, transmitter_am, transmitter_amssb]
}