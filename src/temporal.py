import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

def extract_temporal_features(ig_sequence, window=5):
    """
    Computes statistical 'momentum' for each token.
    ig_sequence: np.ndarray of IG values from Phase 2.
    """
    features = []
    for i in range(len(ig_sequence)):
        # Define look-back window (e.g., last 5 tokens)
        start = max(0, i - window)
        window_ig = ig_sequence[start : i + 1]
        
        # Calculate features
        # Trend is the slope of the line: negative slope = declining grounding
        trend = 0
        if len(window_ig) > 1:
            trend = np.polyfit(range(len(window_ig)), window_ig, 1)[0]
            
        features.append({
            "mean_ig": np.mean(window_ig),
            "std_ig": np.std(window_ig),
            "min_ig": np.min(window_ig),
            "trend": trend,
            "current_ig": ig_sequence[i]
        })
    return features

def plot_ig_with_labels(ig_values, tokens, labels, title="IG_Temporal_Analysis"):
    """
    The 'Money Shot' for your midsem report.
    Shows the IG curve with red blocks over hallucinated tokens.
    """
    # Ensure results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    fig, ax = plt.subplots(figsize=(14, 4))
    x = range(len(ig_values))
    
    # 1. Plot Raw IG (Light blue)
    ax.plot(x, ig_values, alpha=0.3, color='blue', label='Raw IG')
    
    # 2. Plot Smoothed IG (Dark blue) - Using Savitzky-Golay filter
    # This helps see the general trend through the 'noise' of individual tokens
    window_length = min(11, len(ig_values))
    if window_length % 2 == 0: window_length -= 1 # Must be odd
    
    if len(ig_values) > 5:
        smoothed = savgol_filter(ig_values, window_length, 3)
        ax.plot(x, smoothed, color='blue', linewidth=2, label='Smoothed Trend')
    
    # 3. Highlight hallucinated spans (Red background)
    # labels should be binary (1 = hallucination)
    for i in range(min(len(ig_values), len(labels))):
        if labels[i] == 1:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.3, color='red')

    # 4. Formatting
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Information Gain (IG)")
    ax.set_title(title.replace("_", " "))
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/{title}.png", dpi=150)
    print(f"Plot saved to results/{title}.png")
    plt.show()