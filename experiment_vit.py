"""
ViT-based federated learning experiment comparing:
- FedHE-Full: Training from scratch with full encryption
- FedML-HE: Training from scratch with partial encryption (top-10%)
- DictPFL: Starting from PRE-TRAINED model with dictionary compression

This demonstrates DictPFL's advantage: pre-trained initialization + compression.
"""

import numpy as np
import torch
import time

from dataset_cifar import get_federated_cifar10
from model_vit import SimpleViT, get_model_params_vit, set_model_params_vit, train_one_epoch_vit, evaluate_model_vit, load_pretrained_vit
from fhe_utils import FHEContext
from metrics import MetricsTracker, ComparisonMetrics

# Import federated aggregation methods (we'll adapt them for ViT)
from fedhe_full import encrypt_all_gradients, aggregate_encrypted_gradients, decrypt_gradients
from fedml_he import select_topk_gradients, encrypt_partial_gradients, aggregate_partial_encrypted_gradients
from dictpfl import DictPFLManager


def compute_gradients_vit(model_before, model_after, device='cpu'):
    """
    Compute gradients as parameter differences.

    Args:
        model_before: Model before training
        model_after: Model after training
        device: Device for computation

    Returns:
        Dictionary of gradients (parameter differences)
    """
    params_before = get_model_params_vit(model_before)
    params_after = get_model_params_vit(model_after)

    gradients = {}
    for name in params_before.keys():
        gradients[name] = params_after[name] - params_before[name]

    return gradients


def run_vit_experiment(method_name: str, n_rounds: int = 5, n_clients: int = 3,
                      batch_size: int = 64, lr: float = 0.001,
                      use_pretrained: bool = False, pretrain_epochs: int = 3,
                      device='cpu'):
    """
    Run federated learning experiment with ViT.

    Args:
        method_name: 'FedHE-Full', 'FedML-HE', or 'DictPFL'
        n_rounds: Number of federated rounds
        n_clients: Number of clients
        batch_size: Batch size
        lr: Learning rate
        use_pretrained: Whether to start from pre-trained model (DictPFL only)
        pretrain_epochs: Number of pre-training epochs
        device: Device for training

    Returns:
        MetricsTracker
    """
    print(f"\n{'='*70}")
    print(f"Running {method_name}")
    if use_pretrained:
        print(f"Starting from PRE-TRAINED model ({pretrain_epochs} epochs)")
    else:
        print("Training from SCRATCH")
    print(f"{'='*70}")

    # Get data
    client_loaders, test_loader, full_train_loader = get_federated_cifar10(
        n_clients=n_clients, batch_size=batch_size, non_iid=True, alpha=0.5
    )

    # Initialize FHE context
    print("Initializing FHE context...")
    fhe = FHEContext(n=8192, scale=30)

    # Initialize global model
    global_model = SimpleViT(image_size=32, patch_size=4, num_classes=10,
                            dim=128, depth=4, heads=4, mlp_dim=256).to(device)

    # Pre-training for DictPFL
    if use_pretrained:
        global_model = load_pretrained_vit(global_model, pretrain_epochs,
                                          full_train_loader, device)

    # Initialize DictPFL manager if needed
    dictpfl_mgr = None
    if method_name == 'DictPFL':
        print("Initializing DictPFL dictionary decomposition...")
        init_params = get_model_params_vit(global_model)
        dictpfl_mgr = DictPFLManager(rank=32)  # Larger rank for ViT
        dictpfl_mgr.initialize_decomposition(init_params)
        print(f"Dictionary decomposition complete!")

    # Metrics tracker
    tracker = MetricsTracker()

    # Federated training loop
    for round_num in range(1, n_rounds + 1):
        print(f"\nRound {round_num}/{n_rounds}")
        tracker.start_round(round_num)

        # Get global model parameters
        global_params = get_model_params_vit(global_model)

        # Local training on each client
        client_gradients = []
        total_loss = 0.0

        for client_idx, client_loader in enumerate(client_loaders):
            # Create client model
            client_model = SimpleViT(image_size=32, patch_size=4, num_classes=10,
                                    dim=128, depth=4, heads=4, mlp_dim=256).to(device)
            set_model_params_vit(client_model, global_params)

            # Train locally
            loss = train_one_epoch_vit(client_model, client_loader, lr=lr, device=device)
            total_loss += loss

            # Compute gradients
            grads = compute_gradients_vit(global_model, client_model, device)
            client_gradients.append(grads)

        avg_loss = total_loss / n_clients

        # Aggregate gradients based on method
        if method_name == 'FedHE-Full':
            # Encrypt all gradients
            encrypted_grads_list = []
            for client_grads in client_gradients:
                enc_grads, comm_bytes, enc_time = encrypt_all_gradients(client_grads, fhe)
                encrypted_grads_list.append(enc_grads)
                tracker.add_encryption_time(enc_time)
                tracker.add_communication_bytes(comm_bytes)

            # Aggregate
            aggregated_encrypted = aggregate_encrypted_gradients(encrypted_grads_list, n_clients)

            # Decrypt
            original_shapes = {name: grad.shape for name, grad in client_gradients[0].items()}
            aggregated_grads, dec_time = decrypt_gradients(aggregated_encrypted, original_shapes, fhe)
            tracker.add_decryption_time(dec_time)

        elif method_name == 'FedML-HE':
            # Select and encrypt top-k
            client_topk_encrypted = []
            client_remaining_plaintext = []

            for client_grads in client_gradients:
                topk_grads, remaining_grads = select_topk_gradients(client_grads, k_percent=0.1)

                enc_grads, comm_bytes, enc_time = encrypt_partial_gradients(topk_grads, fhe)
                client_topk_encrypted.append(enc_grads)
                client_remaining_plaintext.append(remaining_grads)

                tracker.add_encryption_time(enc_time)
                tracker.add_communication_bytes(comm_bytes)

                # Add plaintext communication
                for grad_array in remaining_grads.values():
                    tracker.add_communication_bytes(grad_array.nbytes)

            # Aggregate
            aggregated_grads, dec_time = aggregate_partial_encrypted_gradients(
                client_topk_encrypted, client_remaining_plaintext, n_clients, fhe
            )
            tracker.add_decryption_time(dec_time)

        elif method_name == 'DictPFL':
            # Use DictPFL with dictionary compression
            from dictpfl import encrypt_lookup_table_gradients, aggregate_encrypted_lookup_tables, decrypt_lookup_table_gradients, prune_gradients

            client_lt_encrypted = []
            lt_shapes = {}

            for client_grads in client_gradients:
                # Convert to lookup table space
                lt_grads = dictpfl_mgr.get_lookup_table_gradients(client_grads)

                # Apply pruning
                pruned_lt_grads = prune_gradients(lt_grads, prune_ratio=0.5, beta=0.2)

                # Encrypt
                enc_grads, comm_bytes, enc_time = encrypt_lookup_table_gradients(pruned_lt_grads, fhe)
                client_lt_encrypted.append(enc_grads)

                tracker.add_encryption_time(enc_time)
                tracker.add_communication_bytes(comm_bytes)

                if not lt_shapes:
                    lt_shapes = {name: grad.shape for name, grad in pruned_lt_grads.items()}

            # Aggregate
            aggregated_encrypted = aggregate_encrypted_lookup_tables(client_lt_encrypted, n_clients)

            # Decrypt
            aggregated_lt_grads, dec_time = decrypt_lookup_table_gradients(
                aggregated_encrypted, lt_shapes, fhe
            )
            tracker.add_decryption_time(dec_time)

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
            updated_global_params[param_name] = global_params[param_name] + aggregated_grads[param_name]
        set_model_params_vit(global_model, updated_global_params)

        # Evaluate
        accuracy = evaluate_model_vit(global_model, test_loader, device)
        tracker.set_accuracy(accuracy)
        tracker.set_loss(avg_loss)

        tracker.end_round()

        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return tracker


def main():
    """Main experiment function."""
    print("="*70)
    print("ViT-based Federated Learning Experiment")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configuration
    n_rounds = 5
    n_clients = 3
    batch_size = 64
    lr = 0.001

    results = {}

    # Experiment 1: FedHE-Full (from scratch)
    results['FedHE-Full'] = run_vit_experiment(
        'FedHE-Full', n_rounds=n_rounds, n_clients=n_clients,
        batch_size=batch_size, lr=lr, use_pretrained=False, device=device
    )

    # Experiment 2: FedML-HE (from scratch)
    results['FedML-HE'] = run_vit_experiment(
        'FedML-HE', n_rounds=n_rounds, n_clients=n_clients,
        batch_size=batch_size, lr=lr, use_pretrained=False, device=device
    )

    # Experiment 3: DictPFL (pre-trained)
    results['DictPFL'] = run_vit_experiment(
        'DictPFL', n_rounds=n_rounds, n_clients=n_clients,
        batch_size=batch_size, lr=lr, use_pretrained=True,
        pretrain_epochs=3, device=device
    )

    # Print results
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)

    for method_name, tracker in results.items():
        print(f"\n{method_name}:")
        tracker.print_summary()

    # Comparison
    comparison = ComparisonMetrics()
    for method_name, tracker in results.items():
        comparison.add_method(method_name, tracker)

    comparison.print_comparison()


if __name__ == "__main__":
    main()
