import time
import os

DEVICE_PATH = '/sys/bus/iio/devices/iio:device3'
if os.name == 'nt':
    DEVICE_PATH = '/tmp/dds_dummy'
    os.makedirs(DEVICE_PATH, exist_ok=True)
else:
    DEVICE_PATH = '/sys/bus/iio/devices/iio:device3'

# Channel files
FREQ1_FILE = 'out_altvoltage0_1A_frequency'
PHASE1_FILE = 'out_altvoltage0_1A_phase'
RAW1_FILE = 'out_altvoltage0_1A_raw'
SCALE1_FILE = 'out_altvoltage0_1A_scale'

FREQ2_FILE = 'out_altvoltage2_2A_frequency'
PHASE2_FILE = 'out_altvoltage2_2A_phase'
RAW2_FILE = 'out_altvoltage2_2A_raw'
SCALE2_FILE = 'out_altvoltage2_2A_scale'

def write_file(filename, value):
    full_path = os.path.join(DEVICE_PATH, filename)
    with open(full_path, 'w') as f:
        f.write(str(value))

def init(phase1=0, phase2_delta=90000, raw1=1, raw2=1):
    """Initialize phase and raw values for both channels."""
    phase2 = phase1 + phase2_delta
    write_file(PHASE2_FILE, phase2)
    write_file(RAW2_FILE, raw2)
    write_file(PHASE1_FILE, phase1)
    write_file(RAW1_FILE, raw1)

def sweep_frequency(start_freq, end_freq, step, delay, scale1=0.177856, scale2=0.177856):
    """Sweep the frequency for both channels."""
    write_file(SCALE1_FILE, scale1)
    write_file(SCALE2_FILE, scale2)
    
    if start_freq <= 0 or end_freq <= 0:
        for _  in range(step):
            time.sleep(delay)
    else:
        for freq in range(start_freq, end_freq + step, step):
            write_file(FREQ2_FILE, freq)
            write_file(FREQ1_FILE, freq)
            time.sleep(delay)

def run_sweep(
    f_start=int(10e6),
    f_step=int(10e6),
    num_steps=50,
    step_duration=0.0001,
    scale1=0.177856,
    scale2=0.177856
):
    """Sweep frequencies starting from f_start. Call init() separately."""
    end_freq = f_start + (num_steps * f_step)
    sweep_frequency(f_start, end_freq, f_step, step_duration, scale1, scale2)

if __name__ == '__main__':
    # Initial configuration (call once before starting sweep)
    init(phase1=0, phase2_delta=90000, raw1=1, raw2=1)
    print("Starting frequency sweep...")
    while True:
        run_sweep()