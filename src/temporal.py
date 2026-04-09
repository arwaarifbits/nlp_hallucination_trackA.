# src/temporal.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind

def extract_temporal_features(ig_sequence: np.ndarray, window: int = 5):
    """
    For each token i, compute features from the preceding window of IG values.
    This captures the "trajectory" of IG leading up to each token.
    """
    features = []
    for i in range(len(ig_sequence)):
        start = max(0, i - window)
        w = ig_sequence[start:i + 1]
        if len(w) >= 2:
            slope = np.polyfit(range(len(w)), w, 1)[0]
        else:
            slope = 0.0
        features.append({
            "position": i,
            "current_ig": ig_sequence[i],
            "mean_ig_window": float(np.mean(w)),
            "std_ig_window": float(np.std(w)),
            "min_ig_window": float(np.min(w)),
            "slope": float(slope),           # negative slope = IG declining = danger signal
        })
    return features

def analyze_precursor_patterns(all_ig_arrays, all_label_arrays, window=5, k=3):
    """
    Key analysis: in the k tokens BEFORE a hallucinated span,
    is the average IG significantly lower than before faithful spans?
    
    all_ig_arrays: list of np.arrays (one per sample)
    all_label_arrays: list of np.arrays of 0/1 labels
    """
    pre_hal_igs = []    # IG values in window before hallucinated tokens
    pre_faith_igs = []  # IG values in window before faithful tokens

    for ig, labels in zip(all_ig_arrays, all_label_arrays):
        min_len = min(len(ig), len(labels))
        ig, labels = ig[:min_len], labels[:min_len]
        
        for i in range(k, len(labels)):
            precursor_mean = np.mean(ig[i-k:i])  # mean IG in k preceding tokens
            if labels[i] == 1:
                pre_hal_igs.append(precursor_mean)
            else:
                pre_faith_igs.append(precursor_mean)

    # Statistical test
    t_stat, p_val = ttest_ind(pre_hal_igs, pre_faith_igs, alternative='less')
    print(f"Mean IG before hallucinated tokens: {np.mean(pre_hal_igs):.4f}")
    print(f"Mean IG before faithful tokens:     {np.mean(pre_faith_igs):.4f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    return pre_hal_igs, pre_faith_igs, p_val

def plot_ig_sequence(ig_values, tokens, labels, sample_id=0, save_dir="results"):
    """
    Plot IG over token positions, shading hallucinated spans in red.
    """
    fig, ax = plt.subplots(figsize=(16, 4))
    x = np.arange(len(ig_values))

    # Raw IG as thin line
    ax.plot(x, ig_values, color='#378ADD', alpha=0.4, linewidth=1, label='Raw IG')

    # Smoothed IG
    if len(ig_values) > 11:
        win = min(11, len(ig_values) // 2 * 2 + 1)
        smoothed = savgol_filter(ig_values, win, 3)
        ax.plot(x, smoothed, color='#185FA5', linewidth=2, label='Smoothed IG')

    # Shade hallucinated regions
    in_span = False
    for i, label in enumerate(labels):
        if label == 1 and not in_span:
            span_start = i
            in_span = True
        elif label == 0 and in_span:
            ax.axvspan(span_start - 0.5, i - 0.5, alpha=0.25, color='#E24B4A',
                      label='Hallucinated' if span_start == min(np.where(np.array(labels)==1)[0]) else "")
            in_span = False
    if in_span:
        ax.axvspan(span_start - 0.5, len(labels) - 0.5, alpha=0.25, color='#E24B4A')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Token labels on x-axis (every 5th to avoid crowding)
    ax.set_xticks(x[::5])
    ax.set_xticklabels([tokens[i].strip() for i in x[::5]], rotation=45, ha='right', fontsize=8)

    ax.set_xlabel("Token position", fontsize=11)
    ax.set_ylabel("Information Gain", fontsize=11)
    ax.set_title(f"Token-level IG with hallucinated spans — sample {sample_id}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ig_temporal_sample_{sample_id}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/ig_temporal_sample_{sample_id}.png")

def plot_precursor_distributions(pre_hal_igs, pre_faith_igs, save_dir="results"):
    """Histogram comparing IG distributions before hallucinated vs faithful tokens."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pre_faith_igs, bins=50, alpha=0.6, color='#378ADD', label='Before faithful tokens', density=True)
    ax.hist(pre_hal_igs, bins=50, alpha=0.6, color='#E24B4A', label='Before hallucinated tokens', density=True)
    ax.set_xlabel("Mean IG in preceding window", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Precursor IG distributions: faithful vs hallucinated", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/precursor_distributions.png", dpi=150)
    plt.close()


def compute_temporal_precedence(all_metric_arrays: dict,
                                 all_label_arrays: list,
                                 window_range=(-3, 1)) -> dict:
    offsets = list(range(window_range[0], window_range[1] + 1))
    metric_names = list(all_metric_arrays.keys())
    
    # --- ADD NORMALIZATION HERE ---
    # We create a normalized copy of the metric arrays so we don't 
    # modify the original data outside this function.
    norm_metric_arrays = {}
    for m_name in metric_names:
        # Flatten all arrays for this metric to get global mean/std
        all_values = np.concatenate(all_metric_arrays[m_name])
        m_mean = np.mean(all_values)
        m_std = np.std(all_values) if np.std(all_values) > 0 else 1.0
        
        # Apply (x - mean) / std to every array in the list
        norm_metric_arrays[m_name] = [
            (arr - m_mean) / m_std for arr in all_metric_arrays[m_name]
        ]
    # ------------------------------

    results = {m: {o: [] for o in offsets} for m in metric_names}
    n_samples = len(all_label_arrays)

    for sample_idx in range(n_samples):
        labels = all_label_arrays[sample_idx]
        in_span = False
        first_hal_positions = []
        for i, lab in enumerate(labels):
            if lab == 1 and not in_span:
                first_hal_positions.append(i)
                in_span = True
            elif lab == 0:
                in_span = False

        for t in first_hal_positions:
            for offset in offsets:
                pos = t + offset
                if pos < 0 or pos >= len(labels):
                    continue
                for m_name in metric_names:
                    # USE THE NORMALIZED ARRAY HERE
                    arr = norm_metric_arrays[m_name][sample_idx]
                    if pos < len(arr):
                        results[m_name][offset].append(arr[pos])

    means = {m: {} for m in metric_names}
    for m in metric_names:
        for o in offsets:
            vals = results[m][o]
            means[m][o] = float(np.mean(vals)) if vals else float('nan')

    return means

def plot_temporal_precedence(means: dict, save_dir="results"):
    """
    Line plot of mean metric value at t-3 to t+1.
    This is the exact plot the rubric requires.
    """
    import matplotlib.pyplot as plt
    
    offsets = sorted(list(list(means.values())[0].keys()))
    x_labels = [f"t{o:+d}" if o != 0 else "t (onset)" for o in offsets]
    
    colors = ['#185FA5', '#E24B4A', '#1D9E75', '#BA7517', '#534AB7']
    
    fig, ax = plt.subplots(figsize=(9, 5))

    for (m_name, offset_dict), color in zip(means.items(), colors):
        y = [offset_dict[o] for o in offsets]
        ax.plot(range(len(offsets)), y, marker='o', linewidth=2,
                markersize=7, label=m_name, color=color)
        
        # Mark the peak
        finite_y = [(i, v) for i, v in enumerate(y) if not np.isnan(v)]
        if finite_y:
            # For hallucination scores: peak = highest value
            peak_i, peak_v = max(finite_y, key=lambda x: x[1])
            ax.annotate(f"peak", xy=(peak_i, peak_v),
                       xytext=(peak_i, peak_v + 0.02),
                       fontsize=8, ha='center', color=color)

    ax.axvline(x=offsets.index(0), color='gray', linestyle='--',
               alpha=0.7, linewidth=1, label='t (first hallucinated token)')
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Position relative to first hallucinated token", fontsize=11)
    ax.set_ylabel("Mean metric score (↑ = hallucination signal)", fontsize=11)
    ax.set_title("Temporal precedence: does signal peak before hallucination onset?", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/temporal_precedence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal precedence plot.")