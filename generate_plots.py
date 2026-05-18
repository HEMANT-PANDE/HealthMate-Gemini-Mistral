import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. Mock Data Generation ---
# Replace this section with your actual data from the 20 test queries.
# The data should be structured with complexity, and the corresponding metrics for each model.
np.random.seed(42) # for reproducible results
num_queries = 20

# Generating mock data that reflects the trends described in your paper
mock_data = {
    # Input complexity (e.g., number of tokens in the prompt)
    'input_complexity': np.linspace(50, 500, num_queries),

    # Gemini: Low and stable latency
    'gemini_latency': np.random.normal(loc=1.8, scale=0.4, size=num_queries),

    # Mistral: Higher latency that increases with complexity
    'mistral_latency': np.linspace(20, 65, num_queries) + np.random.normal(0, 5, num_queries),

    # Gemini: High and stable soundness score
    'gemini_soundness': np.random.normal(loc=1.8, scale=0.1, size=num_queries).clip(0, 2),

    # Mistral: Slightly lower and more variable soundness
    'mistral_soundness': np.random.normal(loc=1.5, scale=0.3, size=num_queries).clip(0, 2)
}
df = pd.DataFrame(mock_data)


# --- 2. Plotting ---

# --- Plot 1: Latency vs. Complexity ---
plt.figure(figsize=(8, 6))
plt.scatter(df['input_complexity'], df['gemini_latency'], color='blue', alpha=0.7, label='Gemini API')
plt.scatter(df['input_complexity'], df['mistral_latency'], color='red', alpha=0.7, label='Mistral CPU')
plt.title('Latency vs. Input Complexity')
plt.xlabel('Input Prompt Complexity (Number of Tokens/Features)')
plt.ylabel('Average Inference Latency (s)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('latency_vs_complexity.png', dpi=300)
plt.show()


# --- Plot 2: Clinical Soundness vs. Complexity ---
plt.figure(figsize=(8, 6))
plt.scatter(df['input_complexity'], df['gemini_soundness'], color='blue', alpha=0.7, label='Gemini API')
plt.scatter(df['input_complexity'], df['mistral_soundness'], color='red', alpha=0.7, label='Mistral CPU')
plt.title('Clinical Soundness vs. Input Complexity')
plt.xlabel('Input Prompt Complexity (Number of Tokens/Features)')
plt.ylabel('Clinical Soundness Score (0-2)')
plt.ylim(0, 2.1) # Set y-axis limit to match the 0-2 scale
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('soundness_vs_complexity.png', dpi=300)
plt.show()
