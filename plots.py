"""
Plotting utilities for visualizing federated learning metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio


def plot_training_metrics(methods_data: Dict[str, Dict], save_path: str = None) -> plt.Figure:
    """
    Plot training metrics comparison across methods.

    Args:
        methods_data: Dictionary mapping method names to their round data
                     Each round data should have keys: 'rounds', 'accuracies', 'losses'
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy over rounds
    ax1 = axes[0]
    for method_name, data in methods_data.items():
        if 'rounds' in data and 'accuracies' in data:
            ax1.plot(data['rounds'], data['accuracies'], marker='o', label=method_name, linewidth=2)

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy over Training Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss over rounds
    ax2 = axes[1]
    for method_name, data in methods_data.items():
        if 'rounds' in data and 'losses' in data:
            ax2.plot(data['rounds'], data['losses'], marker='s', label=method_name, linewidth=2)

    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss over Training Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_efficiency_metrics(methods_data: Dict[str, Dict], save_path: str = None) -> plt.Figure:
    """
    Plot efficiency metrics comparison across methods.

    Args:
        methods_data: Dictionary mapping method names to their round data
                     Each round data should have keys: 'rounds', 'round_times', 'communication_mb'
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Time per round
    ax1 = axes[0]
    for method_name, data in methods_data.items():
        if 'rounds' in data and 'round_times' in data:
            ax1.plot(data['rounds'], data['round_times'], marker='o', label=method_name, linewidth=2)

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Time per Round', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative communication
    ax2 = axes[1]
    for method_name, data in methods_data.items():
        if 'rounds' in data and 'communication_mb' in data:
            cumulative_comm = np.cumsum(data['communication_mb'])
            ax2.plot(data['rounds'], cumulative_comm, marker='s', label=method_name, linewidth=2)

    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Cumulative Communication (MB)', fontsize=12)
    ax2.set_title('Cumulative Communication Cost', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison_bars(summary_data: Dict[str, Dict], save_path: str = None) -> plt.Figure:
    """
    Plot bar chart comparison of summary metrics.

    Args:
        summary_data: Dictionary mapping method names to their summary statistics
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    methods = list(summary_data.keys())
    n_methods = len(methods)

    # Extract metrics
    total_times = [summary_data[m].get('total_time', 0) for m in methods]
    total_comms = [summary_data[m].get('total_communication_mb', 0) for m in methods]
    final_accs = [summary_data[m].get('final_accuracy', 0) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Total time
    ax1 = axes[0]
    bars1 = ax1.bar(methods, total_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:n_methods])
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Total Training Time', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Total communication
    ax2 = axes[1]
    bars2 = ax2.bar(methods, total_comms, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:n_methods])
    ax2.set_ylabel('Communication (MB)', fontsize=12)
    ax2.set_title('Total Communication Cost', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}MB',
                ha='center', va='bottom', fontsize=10)

    # Plot 3: Final accuracy
    ax3 = axes[2]
    bars3 = ax3.bar(methods, final_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:n_methods])
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Final Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.tick_params(axis='x', rotation=45)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_encryption_breakdown(methods_data: Dict[str, Dict], save_path: str = None) -> plt.Figure:
    """
    Plot encryption/decryption time breakdown.

    Args:
        methods_data: Dictionary mapping method names to their round data
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Encryption time per round
    ax1 = axes[0]
    for method_name, data in methods_data.items():
        if 'rounds' in data and 'encryption_times' in data:
            ax1.plot(data['rounds'], data['encryption_times'], marker='o', label=method_name, linewidth=2)

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Encryption Time (seconds)', fontsize=12)
    ax1.set_title('Encryption Time per Round', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Decryption time per round
    ax2 = axes[1]
    for method_name, data in methods_data.items():
        if 'rounds' in data and 'decryption_times' in data:
            ax2.plot(data['rounds'], data['decryption_times'], marker='s', label=method_name, linewidth=2)

    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Decryption Time (seconds)', fontsize=12)
    ax2.set_title('Decryption Time per Round', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Test plotting
    print("Testing plotting utilities...")

    # Generate sample data
    rounds = list(range(1, 11))
    methods_data = {
        'FedHE-Full': {
            'rounds': rounds,
            'accuracies': [0.5 + i * 0.04 for i in range(10)],
            'losses': [1.0 - i * 0.08 for i in range(10)],
            'round_times': [0.5 + np.random.rand() * 0.1 for _ in range(10)],
            'communication_mb': [10.0 + np.random.rand() for _ in range(10)],
            'encryption_times': [0.3 for _ in range(10)],
            'decryption_times': [0.1 for _ in range(10)],
        },
        'FedML-HE': {
            'rounds': rounds,
            'accuracies': [0.52 + i * 0.04 for i in range(10)],
            'losses': [0.98 - i * 0.08 for i in range(10)],
            'round_times': [0.3 + np.random.rand() * 0.05 for _ in range(10)],
            'communication_mb': [5.0 + np.random.rand() * 0.5 for _ in range(10)],
            'encryption_times': [0.15 for _ in range(10)],
            'decryption_times': [0.08 for _ in range(10)],
        },
        'DictPFL': {
            'rounds': rounds,
            'accuracies': [0.54 + i * 0.04 for i in range(10)],
            'losses': [0.96 - i * 0.08 for i in range(10)],
            'round_times': [0.2 + np.random.rand() * 0.03 for _ in range(10)],
            'communication_mb': [2.0 + np.random.rand() * 0.3 for _ in range(10)],
            'encryption_times': [0.08 for _ in range(10)],
            'decryption_times': [0.05 for _ in range(10)],
        },
    }

    # Test training metrics plot
    fig1 = plot_training_metrics(methods_data)
    plt.savefig('/tmp/test_training_metrics.png')
    print("Saved test_training_metrics.png")

    # Test efficiency metrics plot
    fig2 = plot_efficiency_metrics(methods_data)
    plt.savefig('/tmp/test_efficiency_metrics.png')
    print("Saved test_efficiency_metrics.png")

    # Test comparison bars
    summary_data = {
        'FedHE-Full': {'total_time': 5.0, 'total_communication_mb': 100.0, 'final_accuracy': 0.86},
        'FedML-HE': {'total_time': 3.0, 'total_communication_mb': 50.0, 'final_accuracy': 0.88},
        'DictPFL': {'total_time': 2.0, 'total_communication_mb': 20.0, 'final_accuracy': 0.90},
    }
    fig3 = plot_comparison_bars(summary_data)
    plt.savefig('/tmp/test_comparison_bars.png')
    print("Saved test_comparison_bars.png")

    # Test encryption breakdown
    fig4 = plot_encryption_breakdown(methods_data)
    plt.savefig('/tmp/test_encryption_breakdown.png')
    print("Saved test_encryption_breakdown.png")

    print("\nAll plots generated successfully!")
