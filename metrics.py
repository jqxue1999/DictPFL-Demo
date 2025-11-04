"""
Metrics tracking for federated learning experiments.
"""

import time
from typing import Dict, List
import numpy as np


class MetricsTracker:
    """
    Track metrics during federated learning training.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.round_times = []
        self.encryption_times = []
        self.decryption_times = []
        self.communication_bytes = []
        self.accuracies = []
        self.losses = []

        # Per-round breakdown
        self.round_details = []

        # Current round tracking
        self.current_round_start = None
        self.current_round_metrics = {}

    def start_round(self, round_num: int):
        """
        Start tracking a new round.

        Args:
            round_num: Round number
        """
        self.current_round_start = time.time()
        self.current_round_metrics = {
            'round': round_num,
            'encryption_time': 0.0,
            'decryption_time': 0.0,
            'communication_bytes': 0,
            'accuracy': 0.0,
            'loss': 0.0,
        }

    def end_round(self):
        """End tracking current round."""
        if self.current_round_start is not None:
            round_time = time.time() - self.current_round_start
            self.current_round_metrics['total_time'] = round_time

            # Add to historical data
            self.round_times.append(round_time)
            self.encryption_times.append(self.current_round_metrics['encryption_time'])
            self.decryption_times.append(self.current_round_metrics['decryption_time'])
            self.communication_bytes.append(self.current_round_metrics['communication_bytes'])
            self.accuracies.append(self.current_round_metrics['accuracy'])
            self.losses.append(self.current_round_metrics['loss'])

            self.round_details.append(self.current_round_metrics.copy())

            self.current_round_start = None

    def add_encryption_time(self, time_seconds: float):
        """Add encryption time to current round."""
        self.current_round_metrics['encryption_time'] += time_seconds

    def add_decryption_time(self, time_seconds: float):
        """Add decryption time to current round."""
        self.current_round_metrics['decryption_time'] += time_seconds

    def add_communication_bytes(self, num_bytes: int):
        """Add communication bytes to current round."""
        self.current_round_metrics['communication_bytes'] += num_bytes

    def set_accuracy(self, accuracy: float):
        """Set accuracy for current round."""
        self.current_round_metrics['accuracy'] = accuracy

    def set_loss(self, loss: float):
        """Set loss for current round."""
        self.current_round_metrics['loss'] = loss

    def get_summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        if not self.round_times:
            return {
                'total_rounds': 0,
                'total_time': 0.0,
                'avg_round_time': 0.0,
                'total_encryption_time': 0.0,
                'total_decryption_time': 0.0,
                'total_communication_mb': 0.0,
                'final_accuracy': 0.0,
            }

        return {
            'total_rounds': len(self.round_times),
            'total_time': sum(self.round_times),
            'avg_round_time': np.mean(self.round_times),
            'total_encryption_time': sum(self.encryption_times),
            'total_decryption_time': sum(self.decryption_times),
            'total_communication_mb': sum(self.communication_bytes) / (1024 * 1024),
            'final_accuracy': self.accuracies[-1] if self.accuracies else 0.0,
        }

    def get_round_data(self) -> Dict[str, List]:
        """
        Get per-round data for plotting.

        Returns:
            Dictionary with lists of per-round metrics
        """
        return {
            'rounds': list(range(1, len(self.round_times) + 1)),
            'round_times': self.round_times,
            'encryption_times': self.encryption_times,
            'decryption_times': self.decryption_times,
            'communication_mb': [b / (1024 * 1024) for b in self.communication_bytes],
            'accuracies': self.accuracies,
            'losses': self.losses,
        }

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Rounds: {summary['total_rounds']}")
        print(f"Total Time: {summary['total_time']:.2f} seconds")
        print(f"Avg Round Time: {summary['avg_round_time']:.4f} seconds")
        print(f"Total Encryption Time: {summary['total_encryption_time']:.2f} seconds")
        print(f"Total Decryption Time: {summary['total_decryption_time']:.2f} seconds")
        print(f"Total Communication: {summary['total_communication_mb']:.2f} MB")
        print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
        print("="*60)


class ComparisonMetrics:
    """
    Compare metrics across multiple methods.
    """

    def __init__(self):
        """Initialize comparison tracker."""
        self.method_metrics = {}

    def add_method(self, method_name: str, metrics_tracker: MetricsTracker):
        """
        Add metrics for a method.

        Args:
            method_name: Name of the method
            metrics_tracker: MetricsTracker instance
        """
        self.method_metrics[method_name] = metrics_tracker.get_summary()

    def print_comparison(self):
        """Print comparison table."""
        if not self.method_metrics:
            print("No methods to compare.")
            return

        print("\n" + "="*80)
        print("METHOD COMPARISON")
        print("="*80)

        # Print header
        methods = list(self.method_metrics.keys())
        print(f"{'Metric':<30}", end="")
        for method in methods:
            print(f"{method:>15}", end="")
        print()
        print("-"*80)

        # Print metrics
        metrics_to_compare = [
            ('Total Time (s)', 'total_time'),
            ('Avg Round Time (s)', 'avg_round_time'),
            ('Encryption Time (s)', 'total_encryption_time'),
            ('Decryption Time (s)', 'total_decryption_time'),
            ('Communication (MB)', 'total_communication_mb'),
            ('Final Accuracy', 'final_accuracy'),
        ]

        for metric_name, metric_key in metrics_to_compare:
            print(f"{metric_name:<30}", end="")
            for method in methods:
                value = self.method_metrics[method].get(metric_key, 0)
                if metric_key == 'final_accuracy':
                    print(f"{value:>15.4f}", end="")
                elif 'time' in metric_key.lower():
                    print(f"{value:>15.2f}", end="")
                else:
                    print(f"{value:>15.2f}", end="")
            print()

        print("="*80)

    def get_speedup(self, baseline_method: str, comparison_method: str) -> Dict[str, float]:
        """
        Calculate speedup of comparison method vs baseline.

        Args:
            baseline_method: Name of baseline method
            comparison_method: Name of comparison method

        Returns:
            Dictionary with speedup metrics
        """
        if baseline_method not in self.method_metrics or comparison_method not in self.method_metrics:
            return {}

        baseline = self.method_metrics[baseline_method]
        comparison = self.method_metrics[comparison_method]

        speedup = {}
        for key in ['total_time', 'avg_round_time', 'total_encryption_time', 'total_communication_mb']:
            if baseline.get(key, 0) > 0:
                speedup[key] = baseline[key] / comparison[key]
            else:
                speedup[key] = 1.0

        return speedup


if __name__ == "__main__":
    # Test metrics tracker
    tracker = MetricsTracker()

    # Simulate 5 rounds
    for round_num in range(1, 6):
        tracker.start_round(round_num)

        # Simulate metrics
        tracker.add_encryption_time(0.1 + round_num * 0.01)
        tracker.add_decryption_time(0.05 + round_num * 0.005)
        tracker.add_communication_bytes(1024 * 1024 * round_num)
        tracker.set_accuracy(0.5 + round_num * 0.08)
        tracker.set_loss(1.0 - round_num * 0.15)

        time.sleep(0.1)  # Simulate work
        tracker.end_round()

    tracker.print_summary()

    # Test comparison
    comparison = ComparisonMetrics()
    comparison.add_method("Method A", tracker)

    # Create another tracker for comparison
    tracker2 = MetricsTracker()
    for round_num in range(1, 6):
        tracker2.start_round(round_num)
        tracker2.add_encryption_time(0.05)
        tracker2.add_decryption_time(0.02)
        tracker2.add_communication_bytes(512 * 1024)
        tracker2.set_accuracy(0.6 + round_num * 0.07)
        time.sleep(0.05)
        tracker2.end_round()

    comparison.add_method("Method B", tracker2)
    comparison.print_comparison()
