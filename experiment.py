"""
Experiment script to compare FedHE-Full, FedML-HE, and DictPFL methods.
"""

import numpy as np
import torch
import time
from typing import Dict, List

from dataset import get_federated_data
from model import SimpleMLP, train_one_epoch, evaluate_model, get_model_gradients, set_model_params, get_model_params
from fhe_utils import FHEContext
from metrics import MetricsTracker, ComparisonMetrics
from plots import plot_training_metrics, plot_efficiency_metrics, plot_comparison_bars, plot_encryption_breakdown

from fedhe_full import fedhe_full_round
from fedml_he import fedml_he_round
from dictpfl import DictPFLManager, dictpfl_round


def run_federated_training(method_name: str, client_datasets: List, test_data,
                          n_rounds: int = 10, lr: float = 0.01,
                          fhe_context: FHEContext = None,
                          dictpfl_manager: DictPFLManager = None,
                          method_kwargs: Dict = None,
                          init_params: Dict = None) -> MetricsTracker:
    """
    Run federated training with a specific method.

    Args:
        method_name: Name of the method ('FedHE-Full', 'FedML-HE', 'DictPFL')
        client_datasets: List of (X, y) tuples for each client
        test_data: (X_test, y_test) tuple
        n_rounds: Number of training rounds
        lr: Learning rate
        fhe_context: FHE context for encryption
        dictpfl_manager: DictPFL manager (for DictPFL only)
        method_kwargs: Additional method-specific arguments
        init_params: Initial parameters (for fair comparison across methods)

    Returns:
        MetricsTracker with recorded metrics
    """
    if method_kwargs is None:
        method_kwargs = {}

    print(f"\n{'='*60}")
    print(f"Running {method_name}")
    print(f"{'='*60}")

    # Initialize global model with provided initial params if available
    global_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
    if init_params is not None:
        set_model_params(global_model, init_params)
    X_test, y_test = test_data

    # Initialize metrics tracker
    tracker = MetricsTracker()

    # Initialize client models
    n_clients = len(client_datasets)
    client_models = [SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2) for _ in range(n_clients)]

    # Training loop
    for round_num in range(1, n_rounds + 1):
        print(f"\nRound {round_num}/{n_rounds}")
        tracker.start_round(round_num)

        # Step 1: Broadcast global model to clients
        global_params = get_model_params(global_model)
        for client_model in client_models:
            set_model_params(client_model, global_params)

        # Step 2: Local training and gradient computation
        client_gradients = []
        total_loss = 0.0

        for client_idx, (X_client, y_client) in enumerate(client_datasets):
            # Train locally for one epoch
            loss = train_one_epoch(client_models[client_idx], X_client, y_client, lr=lr, batch_size=32)
            total_loss += loss

            # Compute gradients (difference between updated and original params)
            updated_params = get_model_params(client_models[client_idx])
            gradients = {}
            for param_name in global_params.keys():
                gradients[param_name] = updated_params[param_name] - global_params[param_name]

            client_gradients.append(gradients)

        avg_loss = total_loss / n_clients

        # Step 3: Aggregate gradients using the specified method
        if method_name == 'FedHE-Full':
            aggregated_grads, method_metrics = fedhe_full_round(client_gradients, fhe_context)

        elif method_name == 'FedML-HE':
            k_percent = method_kwargs.get('k_percent', 0.1)
            aggregated_grads, method_metrics = fedml_he_round(client_gradients, fhe_context, k_percent=k_percent)

        elif method_name == 'DictPFL':
            prune_ratio = method_kwargs.get('prune_ratio', 0.5)
            beta = method_kwargs.get('beta', 0.2)
            aggregated_grads, method_metrics = dictpfl_round(
                client_gradients, fhe_context, dictpfl_manager, prune_ratio=prune_ratio, beta=beta
            )

        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Update tracker with method metrics
        tracker.add_encryption_time(method_metrics['encryption_time'])
        tracker.add_decryption_time(method_metrics['decryption_time'])
        tracker.add_communication_bytes(method_metrics['communication_bytes'])

        # Step 4: Update global model
        updated_global_params = {}
        for param_name in global_params.keys():
            updated_global_params[param_name] = global_params[param_name] + aggregated_grads[param_name]
        set_model_params(global_model, updated_global_params)

        # Step 5: Evaluate global model
        accuracy = evaluate_model(global_model, X_test, y_test)
        tracker.set_accuracy(accuracy)
        tracker.set_loss(avg_loss)

        tracker.end_round()

        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"  Encryption: {method_metrics['encryption_time']:.4f}s, "
              f"Decryption: {method_metrics['decryption_time']:.4f}s, "
              f"Communication: {method_metrics['communication_bytes']/(1024*1024):.2f} MB")

    return tracker


def main():
    """Main experiment function."""
    print("="*60)
    print("DictPFL Federated Learning Experiment")
    print("="*60)

    # Configuration
    n_clients = 5
    n_samples = 2000
    n_rounds = 10
    lr = 0.01
    seed = 42

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n1. Generating and partitioning dataset...")
    client_datasets = get_federated_data(
        dataset_name='moons',
        n_clients=n_clients,
        n_samples=n_samples,
        non_iid=True,
        alpha=0.5,
        seed=seed
    )

    print(f"Number of clients: {n_clients}")
    for i, (X_client, y_client) in enumerate(client_datasets):
        print(f"  Client {i}: {len(X_client)} samples")

    # Create test dataset (separate from training)
    from dataset import generate_dataset
    X_test, y_test = generate_dataset('moons', n_samples=500, seed=seed+1)
    test_data = (X_test, y_test)

    print("\n2. Initializing FHE context...")
    print("   (This may take a moment...)")
    fhe = FHEContext(n=8192, scale=30)
    print("   FHE context initialized!")

    print("\n3. Creating shared initial model weights...")
    # Create shared initial weights for FAIR comparison
    init_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
    init_params = get_model_params(init_model)
    print(f"   Initial accuracy: {evaluate_model(init_model, X_test, y_test):.4f}")

    print("\n4. Initializing DictPFL manager...")
    dictpfl_mgr = DictPFLManager(rank=8)
    dictpfl_mgr.initialize_decomposition(init_params)
    print("   Dictionary decomposition initialized!")

    # Run experiments for each method with SAME initial weights
    results = {}

    print("\n" + "="*60)
    print("EXPERIMENT 1: FedHE-Full")
    print("="*60)
    tracker_full = run_federated_training(
        method_name='FedHE-Full',
        client_datasets=client_datasets,
        test_data=test_data,
        n_rounds=n_rounds,
        lr=lr,
        fhe_context=fhe,
        init_params=init_params
    )
    results['FedHE-Full'] = tracker_full

    print("\n" + "="*60)
    print("EXPERIMENT 2: FedML-HE")
    print("="*60)
    tracker_partial = run_federated_training(
        method_name='FedML-HE',
        client_datasets=client_datasets,
        test_data=test_data,
        n_rounds=n_rounds,
        lr=lr,
        fhe_context=fhe,
        method_kwargs={'k_percent': 0.1},
        init_params=init_params
    )
    results['FedML-HE'] = tracker_partial

    print("\n" + "="*60)
    print("EXPERIMENT 3: DictPFL")
    print("="*60)
    # Re-initialize DictPFL manager for fresh run
    dictpfl_mgr = DictPFLManager(rank=8)
    dictpfl_mgr.initialize_decomposition(init_params)

    tracker_dictpfl = run_federated_training(
        method_name='DictPFL',
        client_datasets=client_datasets,
        test_data=test_data,
        n_rounds=n_rounds,
        lr=lr,
        fhe_context=fhe,
        dictpfl_manager=dictpfl_mgr,
        method_kwargs={'prune_ratio': 0.5, 'beta': 0.2},
        init_params=init_params
    )
    results['DictPFL'] = tracker_dictpfl

    # Print summaries
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)

    for method_name, tracker in results.items():
        print(f"\n{method_name}:")
        tracker.print_summary()

    # Comparison
    comparison = ComparisonMetrics()
    for method_name, tracker in results.items():
        comparison.add_method(method_name, tracker)

    comparison.print_comparison()

    # Calculate speedups
    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS (vs FedHE-Full)")
    print("="*60)

    for method_name in ['FedML-HE', 'DictPFL']:
        speedup = comparison.get_speedup('FedHE-Full', method_name)
        print(f"\n{method_name}:")
        print(f"  Total Time Speedup: {speedup.get('total_time', 1.0):.2f}x")
        print(f"  Communication Reduction: {speedup.get('total_communication_mb', 1.0):.2f}x")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    methods_data = {}
    summary_data = {}

    for method_name, tracker in results.items():
        methods_data[method_name] = tracker.get_round_data()
        summary_data[method_name] = tracker.get_summary()

    print("Generating training metrics plot...")
    plot_training_metrics(methods_data, save_path='results_training_metrics.png')
    print("  Saved: results_training_metrics.png")

    print("Generating efficiency metrics plot...")
    plot_efficiency_metrics(methods_data, save_path='results_efficiency_metrics.png')
    print("  Saved: results_efficiency_metrics.png")

    print("Generating comparison bars plot...")
    plot_comparison_bars(summary_data, save_path='results_comparison_bars.png')
    print("  Saved: results_comparison_bars.png")

    print("Generating encryption breakdown plot...")
    plot_encryption_breakdown(methods_data, save_path='results_encryption_breakdown.png')
    print("  Saved: results_encryption_breakdown.png")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print("\nAll results and plots have been saved.")


if __name__ == "__main__":
    main()
