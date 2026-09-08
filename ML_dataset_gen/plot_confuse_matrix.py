import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# Use science and ieee styles, force Times New Roman
plt.style.use(['ieee'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titlesize': 8,      # Reduced from 10
    'axes.labelsize': 8,      # Reduced from 10
    'xtick.labelsize': 6,     # Reduced from 8
    'ytick.labelsize': 6,     # Reduced from 8
    'legend.fontsize': 6,     # Reduced from 8
    'font.size': 6            # Reduced from 8 (affects colorbar)
})

# Confusion matrix data
cm = np.array([
    [2.25000000e-01, 5.32000000e-01, 2.42833333e-01, 0.00000000e+00,
     1.66666667e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [1.72833333e-01, 6.23833333e-01, 2.03000000e-01, 1.66666667e-04,
     1.66666667e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [3.63333333e-02, 6.00000000e-03, 9.57666667e-01, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [1.33333333e-03, 5.00000000e-04, 1.30000000e-02, 9.78666667e-01,
     6.00000000e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     3.33333333e-04, 0.00000000e+00, 1.66666667e-04],
    [2.66666667e-03, 6.00000000e-03, 1.33333333e-02, 3.33333333e-04,
     9.77000000e-01, 0.00000000e+00, 0.00000000e+00, 1.66666667e-04,
     1.66666667e-04, 3.33333333e-04, 0.00000000e+00],
    [3.33333333e-04, 6.66666667e-04, 2.91666667e-02, 0.00000000e+00,
     8.33333333e-04, 9.53333333e-01, 0.00000000e+00, 1.48333333e-02,
     0.00000000e+00, 0.00000000e+00, 8.33333333e-04],
    [1.33333333e-03, 1.66666667e-04, 2.58333333e-02, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 9.70333333e-01, 3.33333333e-04,
     0.00000000e+00, 0.00000000e+00, 2.00000000e-03],
    [1.33333333e-03, 5.00000000e-04, 2.78333333e-02, 0.00000000e+00,
     1.66666667e-03, 0.00000000e+00, 0.00000000e+00, 9.64000000e-01,
     1.66666667e-04, 1.66666667e-04, 4.33333333e-03],
    [1.50000000e-03, 1.66666667e-04, 2.36666667e-02, 0.00000000e+00,
     5.00000000e-04, 0.00000000e+00, 1.66666667e-04, 5.33333333e-03,
     9.61833333e-01, 3.83333333e-03, 3.00000000e-03],
    [5.00000000e-04, 6.66666667e-04, 2.51666667e-02, 0.00000000e+00,
     3.33333333e-04, 1.66666667e-04, 1.66666667e-04, 2.83333333e-03,
     1.00000000e-03, 9.29500000e-01, 3.96666667e-02],
    [1.16666667e-03, 1.83333333e-03, 2.55000000e-02, 0.00000000e+00,
     3.33333333e-04, 0.00000000e+00, 0.00000000e+00, 7.00000000e-03,
     0.00000000e+00, 2.66666667e-03, 9.61500000e-01]
])

# Modulation class labels
labels = [
    "WBFM", "AM-DSB", "AM-SSB", "CPFSK", "GFSK",
    "PAM4", "BPSK", "QPSK", "8PSK", "QAM16", "QAM64",
]

# Custom annotation matrix: hide numbers < 0.01 to declutter
annot_labels = np.empty_like(cm, dtype=str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        annot_labels[i, j] = f"{val:.2f}" if val >= 0.01 else ""

# Create the plot
plt.figure(figsize=(5.5, 4.5))
ax = sns.heatmap(
    cm,
    annot=annot_labels,  
    fmt='',              
    cmap='Blues',        
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor='white',   
    annot_kws={"size": 5, "family": "serif"}, # Reduced inside-box font size to 5
    vmin=0,
    vmax=1,
    cbar_kws={'shrink': 0.8}  
)

# Label axes clearly
plt.xlabel('Predicted Modulation', fontweight='bold', labelpad=10)
plt.ylabel('True Modulation', fontweight='bold', labelpad=10)

# Rotate and align tick labels, remove tick lines
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.tick_params(left=False, bottom=False) 

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()