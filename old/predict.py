from old.MarkovChain import *
import numpy as np

idle = State(
    "IDLE",
    power=0.1,
    bandwidth=0.0,
    noise_level=1e-3
)

s1 = State(
    "state 1",
    power=1.0,
    bandwidth=20e6,
    noise_level=5e-3
)

s2 = State(
    "state 2",
    power=2.0,
    bandwidth=5e6,
    noise_level=1e-3
)

s3 = State(
    "state 3",
    power=2.0,
    bandwidth=5e6,
    noise_level=1e-3
)

s4 = State(
    "state 4",
    power=2.0,
    bandwidth=5e6,
    noise_level=1e-3
)
states = [idle, s1, s2, s3]



import matplotlib.pyplot as plt

# Test accuracy vs number of steps
step_counts = range(10, 1001, 50)
accuracies_mean = []
accuracies_std = []

for n_steps in step_counts:
    accuracies_runs = []
    
    for run in range(50):
        P = [[np.random.random() for _ in range(len(states))] for _ in range(len(states))]
        # Normalize rows to sum to 1
        for i in range(len(states)):
            row_sum = sum(P[i])
            P[i] = [x / row_sum for x in P[i]]

        mc = MarkovChain(states, P)
        path = mc.simulate(start_state=idle, n_steps=n_steps)
        transition_count = {}
        
        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i + 1]
            current_index = states.index(current_state)
            next_index = states.index(next_state)
            key = (current_index, next_index)
            transition_count[key] = transition_count.get(key, 0) + 1
        
        # Reconstruct probability matrix
        P_reconstructed = [[0.0] * len(states) for _ in range(len(states))]
        for (from_idx, to_idx), count in transition_count.items():
            P_reconstructed[from_idx][to_idx] = count
        
        # Normalize to get probabilities
        for i in range(len(states)):
            total = sum(P_reconstructed[i])
            if total > 0:
                P_reconstructed[i] = [x / total for x in P_reconstructed[i]]
        
        # Calculate accuracy
        accuracy = np.mean([abs(P_reconstructed[i][j] - P[i][j]) 
                            for i in range(len(states)) 
                            for j in range(len(states))])
        accuracies_runs.append(1 - accuracy)
    
    accuracies_mean.append(np.mean(accuracies_runs))
    accuracies_std.append(np.std(accuracies_runs))

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(step_counts, accuracies_mean, yerr=accuracies_std, marker='o', linewidth=2, capsize=5)
plt.xlabel('Number of Steps')
plt.ylabel('Accuracy')
plt.title('Markov Chain Simulation Accuracy vs Steps')
plt.grid(True)
#plt.xscale('log')
plt.show()
