import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter1d

from sfcw import generate_sfcw
from MarkovChain import State, MarkovChain

idle = State("IDLE",    f_start=0,    f_step=0,    num_steps=64,  step_duration=40e-6)
s1   = State("STATE_1", f_start=10e6,  f_step=30e3, num_steps=80,  step_duration=50e-6)
s2   = State("STATE_2", f_start=10e6,  f_step=30e3, num_steps=60,  step_duration=50e-6)
s3   = State("STATE_3", f_start=10e6,  f_step=30e3, num_steps=112, step_duration=50e-6)

states = [idle, s1, s2, s3]
    
state_labels = [s.name for s in states]

label_to_index = {label: i for i, label in enumerate(state_labels)}

def generate_signal(fs=60e6, snr_db=5, seed=123, steps =400,   
    P = [[0.5, 0.2, 0.1, 0.2], 
         [0.5, 0.3, 0.1, 0.1], 
         [0.7, 0.1, 0.1, 0.1], 
         [0.6, 0.1, 0.1, 0.2]],
    ):
    """1. PARAMETERS & SIGNAL GENERATION"""
    rng = np.random.default_rng(seed)
    
    mc = MarkovChain(states, P)
    path = mc.simulate(start_state=idle, n_steps=steps)
    
    total_sfcw = np.zeros(0, dtype=np.complex128)
    for s in path:
        t, sfcw, _ = generate_sfcw(
            f_start=s.get_var("f_start"), f_step=s.get_var("f_step"),
            num_steps=s.get_var("num_steps"), step_duration=s.get_var("step_duration"),
            fs=fs, amplitude=1.0
        )
        total_sfcw = np.append(total_sfcw, sfcw)
    
    sig_pwr = np.mean(np.abs(total_sfcw)**2)
    noise_std = np.sqrt((sig_pwr / (10**(snr_db/10))) / 2)
    total_sfcw += noise_std * (rng.standard_normal(len(total_sfcw)) + 1j*rng.standard_normal(len(total_sfcw)))
    
    return total_sfcw, path, states, fs


def compute_stft(total_sfcw, fs, win_len=512*2, hop_len=128):
    """2. STFT & RIDGE TRACKING"""
    win = gaussian(win_len, std=win_len/12, sym=True)
    SFT = ShortTimeFFT(win=win, hop=hop_len, fs=fs, fft_mode="centered")
    
    Sxx_mag = np.abs(SFT.stft(total_sfcw))
    f, t_spec = SFT.f, SFT.t(len(total_sfcw))
    
    ridge_idx = np.argmax(Sxx_mag, axis=0)
    ridge_freq = f[ridge_idx]
    ridge_smooth = gaussian_filter1d(ridge_freq, sigma=2)
    # plt.figure()
    # plt.plot(t_spec, ridge_smooth)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    # plt.title("Smoothed Ridge Frequency vs Time")
    # plt.grid(True)
    # plt.show()
    return Sxx_mag, f, t_spec, ridge_smooth


def segment_signal(ridge_smooth, t_spec, plot=True):
    """3. SEGMENTATION VIA FREQUENCY GRADIENT (with optional plotting)"""
    
    df_dt = np.abs(np.gradient(ridge_smooth, t_spec))
    threshold = np.median(df_dt) + 3.0 * np.std(df_dt)
    change_points = np.where(df_dt > threshold)[0]
    
    # Clean nearby change points
    if len(change_points) > 0:
        cleaned = [change_points[0]]
        for p in change_points[1:]:
            if p - cleaned[-1] > 5:
                cleaned.append(p)
        change_points = np.array(cleaned)
    
    segments = np.split(np.arange(len(ridge_smooth)), change_points)
    
    # -------- Plotting inside function --------
    if plot:
        plt.figure()
        
        # Light background ridge
        plt.plot(t_spec, ridge_smooth, alpha=0.3)
        
        # Plot each segment separately
        for seg in segments:
            plt.plot(t_spec[seg], ridge_smooth[seg], linewidth=2)
        
        # Mark change points
        for cp in change_points:
            plt.axvline(t_spec[cp], linestyle='--')
        
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Segmented Ridge")
        plt.grid(True)
        plt.show()
    
    return segments


def classify_segments(segments, ridge_freq, t_spec, states, f_threshold=5e6):
    """4. CLASSIFICATION & ACTIVE BANDWIDTH EXTRACTION"""
    state_params = []
    for s in states:
        state_params.append({
            'name': s.name,
            'bw_mhz': (abs(s.get_var("f_step")) * s.get_var("num_steps")) / 1e6,
            "chrip": ""
        })
    print(state_params)
    print(f"{'Detected':<12} | {'Start (ms)':<10} | {'Dur (ms)':<8} | {'Active BW'}")
    print("-" * 55)
    
    results = []
    for seg in segments:
        
        if len(seg) < 5:
            continue
        
        f_seg = ridge_freq[seg]
        t_seg = t_spec[seg]
        print(f_seg)
        active_indices = np.where(f_seg > f_threshold)[0]
        
        if len(active_indices) < 3:
            label = "IDLE"
            bw = 0.0
            t0_ms = t_seg[0] * 1000
            dur_ms = (t_seg[-1] - t_seg[0]) * 1000
            print(f"{label:<12} | {t0_ms:<10.2f} | {dur_ms:<8.2f} | {bw:<8.2f} MHz")
        else:
            f_active = f_seg[active_indices]
            bw = (np.max(f_active) - np.min(f_active)) / 1e6
            t0_ms = t_seg[active_indices[0]] * 1000
            dur_ms = (t_seg[active_indices[-1]] - t_seg[active_indices[0]]) * 1000
            
            is_linear = np.abs(np.corrcoef(np.arange(len(f_active)), f_active)[0,1])
            
            min_error = float('inf')
            best_match = None
            for sp in state_params:
                if sp['name'] == "IDLE":
                    continue
                bw_error = abs(bw - sp['bw_mhz'])
                if bw_error < min_error:
                    min_error = bw_error
                    best_match = sp
            
            label = best_match['name'] if is_linear > 0.8 and min_error < 0.9 else "IDLE"
            #print(f"{is_linear}|{min_error}")
        
            print(f"{label:<12} | {t0_ms:<10.2f} | {dur_ms:<8.2f} | {bw:<8.2f} MHz|{is_linear}|{min_error}")
        results.append({'t0': t0_ms, 't1': t0_ms + dur_ms, 'label': label})
    
    return results


def plot_results(Sxx_mag, f, t_spec, results, f_threshold=0.0e6):
    """5. PLOTTING"""
    plt.figure(figsize=(12, 6))
    plt.imshow(20*np.log10(Sxx_mag + 1e-12), aspect='auto', origin='lower',
               extent=[t_spec[0]*1e3, t_spec[-1]*1e3, f[0]*1e-6, f[-1]*1e-6], cmap='viridis')
    
    for r in results:
        color = 'red' if r['label'] == "IDLE" else 'white'
        plt.axvspan(r['t0'], r['t1'], color=color, alpha=0.2)
        plt.text(r['t0']+0.1, f[-1]*1e-6*0.9, r['label'], color='white', weight='bold', fontsize=8)
    
    plt.axhline(f_threshold/1e6, color='orange', linestyle='--', alpha=0.5, label='DC Mask')
    plt.title("SFCW Detection: BW Measured Excl. DC-Start")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.colorbar(label="Magnitude (dB)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_with_path(results, path):
    """6. COMPARISON WITH GENERATED PATH"""
    path_consolidate = []

    for s in path:
        if not path_consolidate or s.name != "IDLE" or path_consolidate[-1].name != "IDLE":
            path_consolidate.append(s)
    
    correct_detections = 0
    # for result in results:
    #     detected_label = result['label']
    #     if detected_label in [s.name for s in path_consolidate]:
    #     correct_detections += 1
    
    for i, s in enumerate(path_consolidate, 1):
        detected = results[i-1]['label'] if i-1 < len(results) else 'N/A'
        #print(f"State {i:02d}: Expected: {s.name}, Detected: {detected}")
        if s.name == detected:
            correct_detections += 1
        else:
            print(f"MISMATCH State {i:02d}: Expected: {s.name}, Detected: {detected}")
    
    
    total_states = len(path_consolidate)
    accuracy = (correct_detections / total_states) * 100 if total_states > 0 else 0
    accuracy = accuracy if correct_detections <= total_states else 0
    print(f"Total States: {total_states}, Correctly Detected: {correct_detections}, Accuracy: {accuracy:.2f}%")
    #print expcected vs detected states
    print("\nExpected vs Detected States:")
    
    return accuracy

def reconstruct_transition_matrix(path, label_to_index):
    """
    Reconstruct transition probability matrix from a label-based path.
    """
    print(path)
    transition_count = {}
    
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i + 1]
        current_index = label_to_index[current_state["label"]]
        next_index = label_to_index[next_state["label"]]
        key = (current_index, next_index)
        transition_count[key] = transition_count.get(key, 0) + 1
    print(transition_count)
    
    # Reconstruct probability matrix
    P_reconstructed = [[0.0] * len(states) for _ in range(len(states))]
    for (from_idx, to_idx), count in transition_count.items():
        P_reconstructed[from_idx][to_idx] = count
    
    # Normalize to get probabilities
    for i in range(len(states)):
        total = sum(P_reconstructed[i])
        if total > 0:
            P_reconstructed[i] = [x / total for x in P_reconstructed[i]]

    return P_reconstructed



import multiprocessing as mp
import concurrent.futures
    
def run_single_snr(snr_db):
    accuracies_run = []
    for i in range(3):  # Run each SNR value 3 times
        total_sfcw, path, states, fs = generate_signal(
            seed=1213456216, snr_db=snr_db, steps = 20
        )
        Sxx_mag, f, t_spec, ridge_smooth = compute_stft(total_sfcw, fs)
        segments = segment_signal(ridge_smooth, t_spec)
        results = classify_segments(segments, ridge_smooth, t_spec, states)
        accuracy = compare_with_path(results, path)
        accuracies_run.append(accuracy)
        print(f"SNR: {snr_db} dB, Run {i}, Accuracy: {accuracy:.2f}%")
    return np.mean(accuracies_run), np.std(accuracies_run)
def main():
    
    all_accuracy = []
    
    P = [[0.5, 0.2, 0.1, 0.2], 
    [0.5, 0.3, 0.1, 0.1], 
    [0.7, 0.1, 0.1, 0.1], 
    [0.6, 0.1, 0.1, 0.2]]

    if True:
        for i in range(1):  # Run multiple times to test consistency
            # 1. Generate signal
            total_sfcw, path, states, fs = generate_signal(seed=1213456216, snr_db=10, P=P,steps=20)  # You can change seed and SNR for different runs
            # 2. Compute STFT
            Sxx_mag, f, t_spec, ridge_smooth = compute_stft(total_sfcw, fs)
            # 3. Segment signal
            segments = segment_signal(ridge_smooth, t_spec)
            # # 4. Classify segments
            # results = classify_segments(segments, ridge_smooth, t_spec, states)
            # # 5. Plot results
            # plot_results(Sxx_mag, f, t_spec, results)
            # # 6. Compare with path
            # accuracy = compare_with_path(results, path)
            # print(len(results))
            # all_accuracy.append(accuracy)
            # print(f"Run {i+1}, Detection accuracy: {accuracy:.2f}%")
            
            # Reconstruct transition matrix and calculate accuracy
            # P_reconstructed = reconstruct_transition_matrix(results, label_to_index)
            # print(P)
            # print(P_reconstructed)
            # error = np.mean([abs(P_reconstructed[i][j] - P[i][j]) 
            #             for i in range(len(states)) 
            #             for j in range(len(states))])
            # accuracy = 1 - error
            # print(f"Run {i+1}, reconstruct accuracy : {accuracy:.2f}%")
        
    if False:
        snr_values = range(-10, 10)  # SNR values from -10 to 10 dB
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_cores
        ) as executor:
            results = list(executor.map(run_single_snr, snr_values))
        
        accuracies, std_dev = zip(*results)

        plt.figure(figsize=(10, 5))
        plt.errorbar(snr_values, accuracies, yerr=std_dev, marker='o', capsize=5, fmt='o')
        plt.title('Accuracy vs SNR')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy (%)')
        plt.grid()
        plt.xticks(snr_values)
        plt.ylim(auto=True)
        plt.show()


if __name__ == "__main__":
    main()
    # You can now call main() multiple times