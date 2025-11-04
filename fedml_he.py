"""
FedML-HE: Baseline method that encrypts only top-k% gradients by magnitude.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from fhe_utils import FHEContext, get_ciphertext_size
from Pyfhel import PyCtxt


def select_topk_gradients(gradients: Dict[str, np.ndarray], k_percent: float = 0.1) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Select top-k% gradients by magnitude and separate them from the rest.

    Args:
        gradients: Dictionary mapping parameter names to gradient arrays
        k_percent: Percentage of gradients to encrypt (0.0 to 1.0)

    Returns:
        Tuple of (topk_gradients, remaining_gradients) where each is a dict
    """
    # Flatten all gradients to find global top-k
    all_grads = []
    param_info = []  # (param_name, start_idx, shape)

    for param_name, grad_array in gradients.items():
        flat_grad = grad_array.flatten()
        start_idx = len(all_grads)
        all_grads.extend(flat_grad)
        param_info.append((param_name, start_idx, grad_array.shape))

    all_grads = np.array(all_grads)

    # Find top-k by absolute value
    k = max(1, int(len(all_grads) * k_percent))
    topk_indices = np.argsort(np.abs(all_grads))[-k:]
    topk_mask = np.zeros(len(all_grads), dtype=bool)
    topk_mask[topk_indices] = True

    # Split gradients
    topk_gradients = {}
    remaining_gradients = {}

    for param_name, start_idx, shape in param_info:
        n_elements = np.prod(shape)
        end_idx = start_idx + n_elements

        # Extract masks for this parameter
        param_topk_mask = topk_mask[start_idx:end_idx]

        # Create arrays
        flat_grad = all_grads[start_idx:end_idx]

        flat_topk = np.zeros(n_elements)
        flat_remaining = np.zeros(n_elements)

        flat_topk[param_topk_mask] = flat_grad[param_topk_mask]
        flat_remaining[~param_topk_mask] = flat_grad[~param_topk_mask]

        topk_gradients[param_name] = flat_topk.reshape(shape)
        remaining_gradients[param_name] = flat_remaining.reshape(shape)

    return topk_gradients, remaining_gradients


def encrypt_partial_gradients(topk_gradients: Dict[str, np.ndarray],
                              fhe_context: FHEContext) -> Tuple[Dict[str, List[PyCtxt]], int, float]:
    """
    Encrypt only the top-k gradients.

    Args:
        topk_gradients: Dictionary of top-k gradients to encrypt
        fhe_context: FHE context for encryption

    Returns:
        Tuple of (encrypted_gradients, communication_bytes, encryption_time)
    """
    start_time = time.time()
    encrypted_grads = {}
    total_bytes = 0

    for param_name, grad_array in topk_gradients.items():
        # Only encrypt non-zero elements (sparse representation)
        flat_grad = grad_array.flatten()
        nonzero_indices = np.nonzero(flat_grad)[0]
        nonzero_values = flat_grad[nonzero_indices]

        if len(nonzero_values) > 0:
            # Encrypt the non-zero values
            encrypted_list = fhe_context.encrypt_gradients(nonzero_values)
            encrypted_grads[param_name] = {
                'encrypted': encrypted_list,
                'indices': nonzero_indices,
                'shape': grad_array.shape
            }

            # Track communication size
            for ctxt in encrypted_list:
                total_bytes += get_ciphertext_size(ctxt)

            # Also need to send indices (4 bytes per index)
            total_bytes += len(nonzero_indices) * 4

    encryption_time = time.time() - start_time

    return encrypted_grads, total_bytes, encryption_time


def aggregate_partial_encrypted_gradients(client_encrypted_grads: List[Dict],
                                          client_plaintext_grads: List[Dict[str, np.ndarray]],
                                          n_clients: int,
                                          fhe_context: FHEContext) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Aggregate partial encrypted gradients and plaintext gradients.

    Note: Each client may select different top-k indices, so we need to handle
    sparse gradients from different clients properly.

    Args:
        client_encrypted_grads: List of encrypted top-k gradients from each client
        client_plaintext_grads: List of remaining plaintext gradients from each client
        n_clients: Number of clients
        fhe_context: FHE context

    Returns:
        Tuple of (aggregated_gradients, decryption_time)
    """
    start_time = time.time()

    # Get parameter names
    param_names = client_plaintext_grads[0].keys()
    aggregated = {}

    for param_name in param_names:
        shape = client_plaintext_grads[0][param_name].shape
        aggregated_array = np.zeros(shape)

        # Step 1: Aggregate plaintext gradients from all clients
        for client_plaintext in client_plaintext_grads:
            aggregated_array += client_plaintext[param_name]

        # Step 2: Decrypt and aggregate encrypted gradients from each client
        # Each client may have encrypted different indices
        for client_idx, client_encrypted in enumerate(client_encrypted_grads):
            if param_name in client_encrypted:
                enc_data = client_encrypted[param_name]

                # Decrypt this client's encrypted gradients
                if len(enc_data['encrypted']) > 0:
                    # Concatenate all ciphertexts for this client
                    decrypted_parts = []
                    for ctxt in enc_data['encrypted']:
                        decrypted_parts.append(fhe_context.decrypt(ctxt))

                    # Flatten and get the values
                    decrypted_vals = np.concatenate(decrypted_parts)
                    indices = enc_data['indices']

                    # Add decrypted values at their corresponding indices
                    flat_agg = aggregated_array.flatten()
                    n_vals = min(len(decrypted_vals), len(indices))
                    for i in range(n_vals):
                        if indices[i] < len(flat_agg):
                            flat_agg[indices[i]] += decrypted_vals[i]

                    aggregated_array = flat_agg.reshape(shape)

        # Average by number of clients
        aggregated[param_name] = aggregated_array / n_clients

    decryption_time = time.time() - start_time

    return aggregated, decryption_time


def fedml_he_round(client_gradients: List[Dict[str, np.ndarray]],
                  fhe_context: FHEContext,
                  k_percent: float = 0.1) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Execute one round of FedML-HE (encrypt top-k% gradients).

    Args:
        client_gradients: List of gradient dictionaries from each client
        fhe_context: FHE context
        k_percent: Percentage of gradients to encrypt

    Returns:
        Tuple of (aggregated_gradients, metrics_dict)
    """
    n_clients = len(client_gradients)
    total_encryption_time = 0.0
    total_communication_bytes = 0

    # Step 1: Each client selects top-k and encrypts them
    client_topk_encrypted = []
    client_remaining_plaintext = []

    for client_grads in client_gradients:
        # Select top-k
        topk_grads, remaining_grads = select_topk_gradients(client_grads, k_percent)

        # Encrypt top-k
        encrypted_grads, comm_bytes, enc_time = encrypt_partial_gradients(topk_grads, fhe_context)

        client_topk_encrypted.append(encrypted_grads)
        client_remaining_plaintext.append(remaining_grads)

        total_encryption_time += enc_time
        total_communication_bytes += comm_bytes

        # Plaintext gradients also need to be sent (much cheaper)
        for grad_array in remaining_grads.values():
            total_communication_bytes += grad_array.nbytes

    # Step 2: Server aggregates (both encrypted and plaintext)
    aggregated_grads, decryption_time = aggregate_partial_encrypted_gradients(
        client_topk_encrypted, client_remaining_plaintext, n_clients, fhe_context
    )

    # Communication: server sends back aggregated gradients
    for grad_array in aggregated_grads.values():
        total_communication_bytes += grad_array.nbytes

    # Collect metrics
    metrics = {
        'encryption_time': total_encryption_time,
        'decryption_time': decryption_time,
        'communication_bytes': total_communication_bytes,
    }

    return aggregated_grads, metrics


if __name__ == "__main__":
    # Test FedML-HE
    print("Testing FedML-HE method...")

    # Create FHE context
    print("Initializing FHE context...")
    fhe = FHEContext(n=8192, scale=30)

    # Simulate gradients from 3 clients
    n_clients = 3
    client_grads = []

    for i in range(n_clients):
        grads = {
            'fc1.weight': np.random.randn(32, 2) * 0.1,
            'fc1.bias': np.random.randn(32) * 0.1,
            'fc2.weight': np.random.randn(2, 32) * 0.1,
            'fc2.bias': np.random.randn(2) * 0.1,
        }
        client_grads.append(grads)

    # Run one round
    print("Running FedML-HE round...")
    aggregated, metrics = fedml_he_round(client_grads, fhe, k_percent=0.1)

    print("\n--- Results ---")
    print(f"Encryption time: {metrics['encryption_time']:.4f} seconds")
    print(f"Decryption time: {metrics['decryption_time']:.4f} seconds")
    print(f"Communication: {metrics['communication_bytes'] / (1024*1024):.2f} MB")

    # Verify correctness (compare with plaintext averaging)
    print("\n--- Verification ---")
    for param_name in client_grads[0].keys():
        # Plaintext average
        plaintext_avg = np.mean([grads[param_name] for grads in client_grads], axis=0)

        # Compare with result
        max_error = np.max(np.abs(plaintext_avg - aggregated[param_name]))
        print(f"{param_name}: max error = {max_error:.10f}")
