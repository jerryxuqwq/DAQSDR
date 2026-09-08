#!/usr/bin/env python3
import adi
import numpy as np
import pickle
import random
from gnuradio import gr, blocks
from transmitters import transmitters
from source_alphabet import source_alphabet

# ---------------------------------------------------------
# 1. Hardware Setup (DAQ2)
# ---------------------------------------------------------
NUM_SAMPLES = int(1024 * 1024 / 4 / 4)

print("Initializing DAQ2...")
dev = adi.DAQ2("ip:analog.local")
dev._ctx.set_timeout(3000)
dev._rxadc.set_kernel_buffers_count(1)
dev._txdac.set_kernel_buffers_count(2)
dev.rx_enabled_channels = [0, 1]  # Assuming Ch0=I, Ch1=Q
dev.tx_enabled_channels = [0, 1]
dev.rx_buffer_size = NUM_SAMPLES
dev.tx_cyclic_buffer = False
fs = int(dev.sample_rate)
print("Device Started")

# ---------------------------------------------------------
# 2. Dataset Parameters
# ---------------------------------------------------------
dataset = {}
nvecs_per_key = 1000
vec_length = 128

# Note: In an OTA environment, programmatic SNR is replaced by physical realities.
# We replace the synthetic 'snr_vals' loop with a hardware gain/attenuation loop.
# If your DAQ2 setup lacks programmable attenuation, these serve as dataset labels 
# for different physical ranges or TX power states you manually configure.
gain_states = range(-20, 20, 2) 

for gain_label in gain_states:
    print(f"Current OTA Gain State / Label: {gain_label}")
    
    for alphabet_type in transmitters.keys():
        for i, mod_type in enumerate(transmitters[alphabet_type]):
            dataset[(mod_type.modname, gain_label)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
            
            insufficient_modsnr_vectors = True
            modvec_indx = 0
            
            while insufficient_modsnr_vectors:
                # --- A. Generate Baseband via GNU Radio ---
                tx_len = int(10e3)
                if mod_type.modname == "QAM16": tx_len = int(20e3)
                if mod_type.modname == "QAM64": tx_len = int(30e3)
                
                src = source_alphabet(alphabet_type, tx_len, True)
                mod = mod_type()
                snk = blocks.vector_sink_c()
                
                tb = gr.top_block()
                tb.connect(src, mod, snk) # Channel model removed for OTA
                tb.run()
                
                # Extract ideal baseband IQ data
                tx_baseband = np.array(snk.data(), dtype=np.complex64)
                
                # Ensure baseband matches DAQ2 NUM_SAMPLES buffer size
                if len(tx_baseband) < NUM_SAMPLES:
                    tx_iq_complex = np.pad(tx_baseband, (0, NUM_SAMPLES - len(tx_baseband)), 'constant')
                else:
                    tx_iq_complex = tx_baseband[:NUM_SAMPLES]
                
                # Split complex data into two distinct channels (I and Q)
                amplitude_scale = (2**15) - 1  # 32767
                
                # Split complex data into I and Q, scale amplitude, and cast to 16-bit integers
                tx_i = (np.real(tx_iq_complex) * amplitude_scale).astype(np.int16)
                tx_q = (np.imag(tx_iq_complex) * amplitude_scale).astype(np.int16)

                # Format for DAQ2 (Assuming complex transmission is expected by the driver/wrapper)
                # If DAQ2 expects separate arrays: tx_iq = [np.real(tx_iq), np.imag(tx_iq)]
                
                # --- B. Transmit and Receive OTA ---
                dev.rx_sync_start = "arm"
                dev.tx_sync_start = "arm"
                if not ("arm" == dev.tx_sync_start == dev.rx_sync_start):
                    raise Exception(f"Unexpected SYNC status: TX {dev.tx_sync_start} RX: {dev.rx_sync_start}")
                
                dev.tx_destroy_buffer()
                dev.rx_destroy_buffer()
                dev._rx_init_channels()
                
                # Pass the data as a list of arrays corresponding to Channel 0 and Channel 1
                dev.tx([tx_i, tx_q])
                dev.tx_sync_start = "trigger_manual"
                
                if not ("disarm" == dev.tx_sync_start == dev.rx_sync_start):
                    raise Exception(f"Unexpected SYNC status: TX {dev.tx_sync_start} RX: {dev.rx_sync_start}")
                
                try:
                    rx_data = dev.rx()
                    # Reconstruct complex signal from raw dual-channel ADC output
                    raw_output_vector = rx_data[0] + 1j * rx_data[1] 
                except Exception as e:
                    print(f"Hardware Transaction FAILED: {e}")
                    continue
                while True:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(12, 4))
                    plt.plot(np.real(raw_output_vector[:1000]))
                    plt.plot(np.imag(raw_output_vector[:1000]))
                    plt.xlabel('Sample')
                    plt.ylabel('Amplitude')
                    plt.title(f'Received Data - {mod_type.modname} at Gain {gain_label}')
                    plt.legend(['I', 'Q'])
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(f'rx_data_{mod_type.modname}_{gain_label}.png', dpi=300, bbox_inches='tight')

                # --- C. Slice and Format Received Data ---
                # Random index start mitigates hardware phase/delay offsets 
                sampler_indx = random.randint(50, 500) 
                
                while sampler_indx + vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
                    sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]
                    
                    # Normalize energy
                    energy = np.sum((np.abs(sampled_vector)))
                    if energy == 0: energy = 1e-10 # Prevent division by zero
                    sampled_vector = sampled_vector / energy
                    
                    dataset[(mod_type.modname, gain_label)][modvec_indx, 0, :] = np.real(sampled_vector)
                    dataset[(mod_type.modname, gain_label)][modvec_indx, 1, :] = np.imag(sampled_vector)
                    
                    sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
                    modvec_indx += 1

                if modvec_indx >= nvecs_per_key:
                    insufficient_modsnr_vectors = False

print("OTA generation complete. Writing to disk...")
with open("OTA_RML_dict.dat", "wb") as f:
    pickle.dump(dataset, f)