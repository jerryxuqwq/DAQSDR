import numpy as np
import rich as print
from rich.traceback import install
install(show_locals=True) # show_locals displays local variables for enhanced debugging


from sfcw import generate_sfcw
from MarkovChain import *
from dds_sweep import init, run_sweep

# ============================================================
# PARAMETERS
# ============================================================

fs = 10e6          # Sampling rate (Hz)
snr_db = 0        # SNR (dB)
rng = np.random.default_rng(123)

# ============================================================
# MARKOV STATES (SFCW MODES)
# ============================================================

idle = State(
    "IDLE",
    f_start=0,
    f_step=0,
    num_steps=64,
    step_duration=0.0001
)

s1 = State(
    "STATE_1",
    f_start=2e6,
    f_step=30e3,
    num_steps=80,
    step_duration=0.0001
)

s2 = State(
    "STATE_2",
    f_start=3e6,
    f_step=35e3,
    num_steps=60,
    step_duration=0.0001
)

s3 = State(
    "STATE_3",
    f_start=4e6,
    f_step=40e3,
    num_steps=112,
    step_duration=0.0001
)

states = [idle, s1, s2, s3]

P = [
    [0, 0.2, 0.2, 0.1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
]
mc = MarkovChain(states, P)

# ============================================================
# GENERATE COMPOSITE SFCW SIGNAL
# ============================================================

total_sfcw = np.zeros(0, dtype=np.complex128)
total_t = np.zeros(0)

path = mc.simulate(start_state=idle, n_steps=50)

init()

for s in path:
    f_start = s.get_var("f_start")
    f_step = s.get_var("f_step")
    num_steps = s.get_var("num_steps")
    step_duration = s.get_var("step_duration")

    run_sweep(
        f_start=int(f_start),
        f_step=int(f_step),
        num_steps=num_steps,
        step_duration=step_duration,
    )
