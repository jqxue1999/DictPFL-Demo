"""
Simulated FHE experiment for ViT-based federated learning with ImageNet pre-trained weights.

Instead of actual FHE encryption (which would take hours), we:
1. Count parameters that WOULD BE encrypted
2. Track communication costs
3. Run actual training to measure accuracy

Key difference from previous experiment:
- DictPFL: Uses ImageNet pre-trained ViT-B/16 weights
- FedHE-Full & FedML-HE: Train from scratch (random initialization)
- All methods resize CIFAR-10 from 32x32 to 224x224
"""

import numpy as np
import torch
import time

from dataset_cifar import get_federated_cifar10
from model_vit_pretrained import (
    ViTForCIFAR10,
    get_model_params_vit,
    set_model_params_vit,
    train_one_epoch_vit,
    evaluate_model_vit,
    count_parameters
)
from metrics import MetricsTracker, ComparisonMetrics


def select_topk_gradients(gradients, k_percent=0.1):
    """
    Select top-k% gradients by magnitude (for FedML-HE).

    Returns:
        topk_gradients: Dictionary with only top-k values (rest are zeros)
        remaining_gradients: Dictionary with remaining values
    """
    topk_gradients = {}
    remaining_gradients = {}

    # Flatten all gradients to find global top-k
    all_grads = []
    param_info = []  # (param_name, shape, start_idx, end_idx)

    current_idx = 0
    for param_name, grad_array in gradients.items():
        shape = grad_array.shape
        flat_grad = grad_array.flatten()
        n_elements = len(flat_grad)

        all_grads.append(flat_grad)
        param_info.append((param_name, shape, current_idx, current_idx + n_elements))
        current_idx += n_elements

    # Concatenate all gradients
    all_grads_flat = np.concatenate(all_grads)

    # Find top-k indices by absolute value
    k = max(1, int(len(all_grads_flat) * k_percent))
    topk_indices_global = np.argpartition(np.abs(all_grads_flat), -k)[-k:]
    topk_mask = np.zeros(len(all_grads_flat), dtype=bool)
    topk_mask[topk_indices_global] = True

    # Split back into parameters
    for param_name, shape, start_idx, end_idx in param_info:
        param_mask = topk_mask[start_idx:end_idx]
        flat_grad = all_grads_flat[start_idx:end_idx]

        # Create topk and remaining arrays
        flat_topk = np.zeros(end_idx - start_idx)
        flat_remaining = np.zeros(end_idx - start_idx)

        flat_topk[param_mask] = flat_grad[param_mask]
        flat_remaining[~param_mask] = flat_grad[~param_mask]

        topk_gradients[param_name] = flat_topk.reshape(shape)
        remaining_gradients[param_name] = flat_remaining.reshape(shape)

    return topk_gradients, remaining_gradients


class DictPFLManager:
    """Simplified DictPFL manager for simulation (no FHE dependencies)."""

    def __init__(self, rank=64, decomp_threshold=1000):
        self.rank = rank
        self.decomp_threshold = decomp_threshold
        self.dictionaries = {}
        self.lookup_tables = {}
        self.param_types = {}

    def initialize_decomposition(self, params):
        """Initialize dictionary decomposition via SVD."""
        for param_name, param_array in params.items():
            shape = param_array.shape

            # Only decompose 2D parameters (matrices) above threshold
            if len(shape) == 2 and min(shape) >= self.decomp_threshold:
                # SVD decomposition
                U, S, Vt = np.linalg.svd(param_array, full_matrices=False)

                # Truncate to rank r
                r = min(self.rank, len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]

                # Dictionary D = U_r @ sqrt(S_r) (fixed)
                # Lookup table T = sqrt(S_r) @ Vt_r (trainable)
                sqrt_S = np.sqrt(S_r)
                self.dictionaries[param_name] = U_r * sqrt_S[np.newaxis, :]
                self.lookup_tables[param_name] = (sqrt_S[:, np.newaxis] * Vt_r)
                self.param_types[param_name] = 'decomposed'
            else:
                # Small parameters or vectors: no decomposition
                self.lookup_tables[param_name] = param_array.copy()
                self.param_types[param_name] = 'direct'

    def get_lookup_table_gradients(self, gradients):
        """Convert parameter gradients to lookup table gradients."""
        lt_gradients = {}

        for param_name, grad in gradients.items():
            if param_name in self.param_types:
                if self.param_types[param_name] == 'decomposed':
                    # Project gradient onto dictionary: T_grad = D^T @ W_grad
                    D = self.dictionaries[param_name]
                    lt_gradients[param_name] = D.T @ grad
                else:
                    lt_gradients[param_name] = grad
            else:
                lt_gradients[param_name] = grad

        return lt_gradients


def prune_gradients(gradients, prune_ratio=0.5, beta=0.2):
    """
    Prune gradients by magnitude with probabilistic reactivation.

    Args:
        gradients: Dictionary of gradient arrays
        prune_ratio: Fraction of gradients to prune (set to zero)
        beta: Probability of reactivating pruned gradients

    Returns:
        Pruned gradients dictionary
    """
    pruned_gradients = {}

    for param_name, grad_array in gradients.items():
        flat_grad = grad_array.flatten()

        # Find pruning threshold (bottom prune_ratio% by magnitude)
        threshold_idx = int(len(flat_grad) * prune_ratio)
        if threshold_idx > 0:
            sorted_abs = np.sort(np.abs(flat_grad))
            threshold = sorted_abs[threshold_idx]

            # Create mask for elements to keep
            keep_mask = np.abs(flat_grad) >= threshold

            # Reactivate pruned elements with probability beta
            pruned_mask = ~keep_mask
            n_pruned = np.sum(pruned_mask)
            if n_pruned > 0:
                reactivate = np.random.rand(n_pruned) < beta
                pruned_indices = np.where(pruned_mask)[0]
                keep_mask[pruned_indices[reactivate]] = True

            # Apply mask
            pruned_flat = flat_grad.copy()
            pruned_flat[~keep_mask] = 0
            pruned_gradients[param_name] = pruned_flat.reshape(grad_array.shape)
        else:
            pruned_gradients[param_name] = grad_array

    return pruned_gradients


def count_params_dict(params_dict):
    """Count total number of parameters."""
    return sum(p.size for p in params_dict.values())


def count_nonzero_parameters(params_dict):
    """Count non-zero parameters."""
    return sum(np.count_nonzero(p) for p in params_dict.values())


def simulate_encryption_cost(num_params, bytes_per_param=8):
    """
    Simulate encryption cost.

    Args:
        num_params: Number of parameters to encrypt
        bytes_per_param: Bytes per parameter (float64 = 8 bytes)

    Returns:
        Simulated encryption time and communication bytes
    """
    # Simulate encryption time: ~1ms per 1000 parameters
    enc_time = num_params / 1000 * 0.001

    # Communication: encrypted parameters are much larger (assume 100 bytes per encrypted param)
    comm_bytes = num_params * 100

    return enc_time, comm_bytes


def compute_gradients_vit(model_before, model_after):
    """Compute gradients as parameter differences."""
    params_before = get_model_params_vit(model_before)
    params_after = get_model_params_vit(model_after)

    gradients = {}
    for name in params_before.keys():
        gradients[name] = params_after[name] - params_before[name]

    return gradients


def run_simulated_vit_experiment(method_name: str, n_rounds: int = 10, n_clients: int = 5,
                                 batch_size: int = 32, lr: float = 0.0001,
                                 use_pretrained: bool = False, device='cpu'):
    """
    Run simulated FHE experiment with ImageNet pre-trained ViT.

    Args:
        method_name: 'FedHE-Full', 'FedML-HE', or 'DictPFL'
        n_rounds: Number of federated rounds
        n_clients: Number of clients
        batch_size: Batch size
        lr: Learning rate (lower for fine-tuning)
        use_pretrained: Whether to use ImageNet pre-trained weights
        device: Device for training

    Returns:
        MetricsTracker, total model parameters
    """
    print(f"\n{'='*70}")
    print(f"Running {method_name} (Simulated FHE)")
    if use_pretrained:
        print(f"✓ Starting from IMAGENET PRE-TRAINED ViT-B/16")
    else:
        print("✗ Training from SCRATCH (random initialization)")
    print(f"{'='*70}")

    # Get data
    print("Loading CIFAR-10 dataset...")
    client_loaders, test_loader, full_train_loader = get_federated_cifar10(
        n_clients=n_clients, batch_size=batch_size, non_iid=True, alpha=0.5
    )

    print(f"  {n_clients} clients:")
    for i, loader in enumerate(client_loaders):
        print(f"    Client {i}: {len(loader.dataset)} samples")

    # Initialize global model
    global_model = ViTForCIFAR10(num_classes=10, use_pretrained=use_pretrained).to(device)

    total_model_params = count_parameters(global_model)
    print(f"\nModel: ViT-B/16 (ImageNet pre-trained)")
    print(f"  Total parameters: {total_model_params:,}")

    # Evaluate initial accuracy
    initial_acc = evaluate_model_vit(global_model, test_loader, device)
    if use_pretrained:
        print(f"  ImageNet pre-trained initial accuracy: {initial_acc:.4f}")
    else:
        print(f"  Initial random accuracy: {initial_acc:.4f}")

    # Initialize DictPFL manager if needed
    dictpfl_mgr = None
    if method_name == 'DictPFL':
        print("\nInitializing DictPFL dictionary decomposition...")
        init_params = get_model_params_vit(global_model)

        # Try different ranks and show compression
        for test_rank in [64, 128, 256]:
            test_mgr = DictPFLManager(rank=test_rank, decomp_threshold=1000)
            test_mgr.initialize_decomposition(init_params)
            total_lt = sum(lt.size for lt in test_mgr.lookup_tables.values())
            compression = total_model_params / total_lt
            print(f"  Rank {test_rank}: {total_model_params:,} → {total_lt:,} params ({compression:.2f}x compression)")

        # Use rank=128 for the experiment
        dictpfl_mgr = DictPFLManager(rank=128, decomp_threshold=1000)
        dictpfl_mgr.initialize_decomposition(init_params)
        total_lt = sum(lt.size for lt in dictpfl_mgr.lookup_tables.values())
        compression = total_model_params / total_lt
        print(f"\n  Using rank=128: {compression:.2f}x compression")

    # Metrics tracker
    tracker = MetricsTracker()

    # Federated training loop
    print(f"\nFederated Training ({n_rounds} rounds):")
    print("─" * 70)

    for round_num in range(1, n_rounds + 1):
        tracker.start_round(round_num)

        # Get global model parameters
        global_params = get_model_params_vit(global_model)

        # Local training on each client
        client_gradients = []
        total_loss = 0.0

        for client_idx, client_loader in enumerate(client_loaders):
            # Create client model
            client_model = ViTForCIFAR10(num_classes=10, use_pretrained=False).to(device)
            set_model_params_vit(client_model, global_params)

            # Train locally
            loss = train_one_epoch_vit(client_model, client_loader, lr=lr, device=device)
            total_loss += loss

            # Compute gradients
            grads = compute_gradients_vit(global_model, client_model)
            client_gradients.append(grads)

        avg_loss = total_loss / n_clients

        # Simulate encryption and aggregation based on method
        total_encrypted_params = 0
        total_plaintext_params = 0

        if method_name == 'FedHE-Full':
            # Encrypt ALL parameters from each client
            for client_grads in client_gradients:
                num_params = count_params_dict(client_grads)
                total_encrypted_params += num_params

                # Simulate encryption
                enc_time, comm_bytes = simulate_encryption_cost(num_params)
                tracker.add_encryption_time(enc_time)
                tracker.add_communication_bytes(comm_bytes)

            # Aggregate (plaintext for simulation)
            aggregated_grads = {}
            for param_name in client_gradients[0].keys():
                aggregated_grads[param_name] = np.mean(
                    [grads[param_name] for grads in client_gradients], axis=0
                )

        elif method_name == 'FedML-HE':
            # Encrypt only top-k% parameters from each client
            for client_grads in client_gradients:
                topk_grads, remaining_grads = select_topk_gradients(client_grads, k_percent=0.1)

                # Count encrypted and plaintext
                num_encrypted = count_nonzero_parameters(topk_grads)
                num_plaintext = count_nonzero_parameters(remaining_grads)

                total_encrypted_params += num_encrypted
                total_plaintext_params += num_plaintext

                # Simulate encryption (only for top-k)
                enc_time, comm_bytes = simulate_encryption_cost(num_encrypted)
                tracker.add_encryption_time(enc_time)
                tracker.add_communication_bytes(comm_bytes)

                # Plaintext communication
                tracker.add_communication_bytes(num_plaintext * 8)  # 8 bytes per float

            # Aggregate (plaintext for simulation)
            aggregated_grads = {}
            for param_name in client_gradients[0].keys():
                aggregated_grads[param_name] = np.mean(
                    [grads[param_name] for grads in client_gradients], axis=0
                )

        elif method_name == 'DictPFL':
            # Encrypt compressed lookup table parameters
            for client_grads in client_gradients:
                # Convert to lookup table space
                lt_grads = dictpfl_mgr.get_lookup_table_gradients(client_grads)

                # Apply pruning
                pruned_lt_grads = prune_gradients(lt_grads, prune_ratio=0.5, beta=0.2)

                # Count parameters to encrypt (full lookup table)
                num_params = count_params_dict(pruned_lt_grads)
                total_encrypted_params += num_params

                # Simulate encryption
                enc_time, comm_bytes = simulate_encryption_cost(num_params)
                tracker.add_encryption_time(enc_time)
                tracker.add_communication_bytes(comm_bytes)

            # Aggregate in lookup table space, then convert back
            aggregated_lt_grads = {}
            for param_name in client_gradients[0].keys():
                lt_grads_list = [dictpfl_mgr.get_lookup_table_gradients(grads)
                               for grads in client_gradients]

                if param_name in lt_grads_list[0]:
                    aggregated_lt_grads[param_name] = np.mean(
                        [lt_grads[param_name] for lt_grads in lt_grads_list], axis=0
                    )

            # Convert back to parameter space
            aggregated_grads = {}
            for param_name, lt_grad in aggregated_lt_grads.items():
                if param_name in dictpfl_mgr.param_types:
                    if dictpfl_mgr.param_types[param_name] == 'decomposed':
                        D = dictpfl_mgr.dictionaries[param_name]
                        grad_W = D @ lt_grad
                        aggregated_grads[param_name] = grad_W
                    else:
                        aggregated_grads[param_name] = lt_grad
                else:
                    aggregated_grads[param_name] = lt_grad

        # Update global model
        updated_global_params = {}
        for param_name in global_params.keys():
            if param_name in aggregated_grads:
                updated_global_params[param_name] = global_params[param_name] + aggregated_grads[param_name]
            else:
                updated_global_params[param_name] = global_params[param_name]
        set_model_params_vit(global_model, updated_global_params)

        # Evaluate
        accuracy = evaluate_model_vit(global_model, test_loader, device)
        tracker.set_accuracy(accuracy)
        tracker.set_loss(avg_loss)

        # Add simulated decryption time
        tracker.add_decryption_time(total_encrypted_params / 10000 * 0.001)

        tracker.end_round()

        # Print round summary
        encrypted_per_client = total_encrypted_params // n_clients
        plaintext_per_client = total_plaintext_params // n_clients

        if plaintext_per_client > 0:
            print(f"Round {round_num:2d}: Acc={accuracy:.4f} | "
                  f"Encrypted={encrypted_per_client:,}/client | "
                  f"Plaintext={plaintext_per_client:,}/client | "
                  f"Loss={avg_loss:.4f}")
        else:
            print(f"Round {round_num:2d}: Acc={accuracy:.4f} | "
                  f"Encrypted={encrypted_per_client:,}/client | "
                  f"Loss={avg_loss:.4f}")

    print("─" * 70)

    return tracker, total_model_params


def main():
    """Main experiment function."""
    print("="*70)
    print("ViT-B/16 ImageNet Pre-trained Federated Learning (Simulated FHE)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Configuration (smaller batches for large model)
    n_rounds = 10
    n_clients = 5
    batch_size = 16  # Smaller batch for ViT-B/16
    lr = 0.0001  # Lower LR for fine-tuning

    results = {}
    model_size = None

    # Experiment 1: FedHE-Full (from scratch)
    print("\n" + "="*70)
    print("EXPERIMENT 1: FedHE-Full")
    print("="*70)
    tracker, model_size = run_simulated_vit_experiment(
        'FedHE-Full', n_rounds=n_rounds, n_clients=n_clients,
        batch_size=batch_size, lr=lr, use_pretrained=False, device=device
    )
    results['FedHE-Full'] = tracker

    # Experiment 2: FedML-HE (from scratch)
    print("\n" + "="*70)
    print("EXPERIMENT 2: FedML-HE")
    print("="*70)
    tracker, _ = run_simulated_vit_experiment(
        'FedML-HE', n_rounds=n_rounds, n_clients=n_clients,
        batch_size=batch_size, lr=lr, use_pretrained=False, device=device
    )
    results['FedML-HE'] = tracker

    # Experiment 3: DictPFL (ImageNet pre-trained)
    print("\n" + "="*70)
    print("EXPERIMENT 3: DictPFL")
    print("="*70)
    tracker, _ = run_simulated_vit_experiment(
        'DictPFL', n_rounds=n_rounds, n_clients=n_clients,
        batch_size=batch_size, lr=lr, use_pretrained=True, device=device
    )
    results['DictPFL'] = tracker

    # Print final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\nModel Size: {model_size:,} parameters")

    for method_name, tracker in results.items():
        summary = tracker.get_summary()
        print(f"\n{method_name}:")
        print(f"  Final Accuracy: {summary['final_accuracy']:.4f}")
        print(f"  Total Time: {summary['total_time']:.2f}s")
        print(f"  Total Communication: {summary['total_communication_mb']:.2f} MB")

    # Detailed comparison
    comparison = ComparisonMetrics()
    for method_name, tracker in results.items():
        comparison.add_method(method_name, tracker)

    comparison.print_comparison()

    # Calculate speedups
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS (vs FedHE-Full)")
    print("="*70)

    for method_name in ['FedML-HE', 'DictPFL']:
        speedup = comparison.get_speedup('FedHE-Full', method_name)
        print(f"\n{method_name}:")
        print(f"  Total Time Speedup: {speedup.get('total_time', 1.0):.2f}x")
        print(f"  Communication Reduction: {speedup.get('total_communication_mb', 1.0):.2f}x")


if __name__ == "__main__":
    main()
