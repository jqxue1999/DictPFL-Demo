"""
FedHE-Full: Baseline method that encrypts all gradients.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from fhe_utils import FHEContext, get_ciphertext_size
from Pyfhel import PyCtxt


def encrypt_all_gradients(gradients: Dict[str, np.ndarray], fhe_context: FHEContext) -> Tuple[Dict[str, List[PyCtxt]], int, float]:
    """
    Encrypt all gradient parameters.

    Args:
        gradients: Dictionary mapping parameter names to gradient arrays
        fhe_context: FHE context for encryption

    Returns:
        Tuple of (encrypted_gradients, communication_bytes, encryption_time)
    """
    start_time = time.time()
    encrypted_grads = {}
    total_bytes = 0

    for param_name, grad_array in gradients.items():
        # Encrypt the gradient
        encrypted_list = fhe_context.encrypt_gradients(grad_array)
        encrypted_grads[param_name] = encrypted_list

        # Track communication size
        for ctxt in encrypted_list:
            total_bytes += get_ciphertext_size(ctxt)

    encryption_time = time.time() - start_time

    return encrypted_grads, total_bytes, encryption_time


def aggregate_encrypted_gradients(client_encrypted_grads: List[Dict[str, List[PyCtxt]]],
                                  n_clients: int) -> Dict[str, List[PyCtxt]]:
    """
    Homomorphically aggregate encrypted gradients from multiple clients.

    Args:
        client_encrypted_grads: List of encrypted gradients from each client
        n_clients: Number of clients (for averaging)

    Returns:
        Aggregated encrypted gradients
    """
    if not client_encrypted_grads:
        return {}

    # Initialize with first client's structure
    aggregated = {}
    param_names = client_encrypted_grads[0].keys()

    for param_name in param_names:
        # Get number of ciphertexts for this parameter
        n_ctxts = len(client_encrypted_grads[0][param_name])

        aggregated[param_name] = []

        # Aggregate each ciphertext position
        for ctxt_idx in range(n_ctxts):
            # Collect ciphertexts from all clients at this position
            ctxts_to_add = [
                client_grads[param_name][ctxt_idx]
                for client_grads in client_encrypted_grads
            ]

            # Homomorphic addition
            aggregated_ctxt = ctxts_to_add[0].copy()
            for ctxt in ctxts_to_add[1:]:
                aggregated_ctxt += ctxt

            # Average by dividing by number of clients (multiply by 1/n)
            # Note: CKKS supports scalar multiplication
            aggregated_ctxt *= (1.0 / n_clients)

            aggregated[param_name].append(aggregated_ctxt)

    return aggregated


def decrypt_gradients(encrypted_grads: Dict[str, List[PyCtxt]],
                     original_shapes: Dict[str, tuple],
                     fhe_context: FHEContext) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Decrypt aggregated gradients.

    Args:
        encrypted_grads: Encrypted gradients
        original_shapes: Original shapes of each parameter
        fhe_context: FHE context for decryption

    Returns:
        Tuple of (decrypted_gradients, decryption_time)
    """
    start_time = time.time()
    decrypted_grads = {}

    for param_name, encrypted_list in encrypted_grads.items():
        shape = original_shapes[param_name]
        decrypted_array = fhe_context.decrypt_gradients(encrypted_list, shape)
        decrypted_grads[param_name] = decrypted_array

    decryption_time = time.time() - start_time

    return decrypted_grads, decryption_time


def fedhe_full_round(client_gradients: List[Dict[str, np.ndarray]],
                    fhe_context: FHEContext) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Execute one round of FedHE-Full (encrypt all gradients).

    Args:
        client_gradients: List of gradient dictionaries from each client
        fhe_context: FHE context

    Returns:
        Tuple of (aggregated_gradients, metrics_dict)
    """
    n_clients = len(client_gradients)
    total_encryption_time = 0.0
    total_communication_bytes = 0

    # Step 1: Each client encrypts their gradients
    encrypted_client_grads = []
    original_shapes = {}

    for client_grads in client_gradients:
        encrypted_grads, comm_bytes, enc_time = encrypt_all_gradients(client_grads, fhe_context)
        encrypted_client_grads.append(encrypted_grads)
        total_encryption_time += enc_time
        total_communication_bytes += comm_bytes

        # Store shapes (same for all clients)
        if not original_shapes:
            for param_name, grad_array in client_grads.items():
                original_shapes[param_name] = grad_array.shape

    # Step 2: Server aggregates encrypted gradients homomorphically
    aggregated_encrypted = aggregate_encrypted_gradients(encrypted_client_grads, n_clients)

    # Communication: server sends back aggregated gradients (same size as one client's upload)
    total_communication_bytes += total_communication_bytes // n_clients

    # Step 3: Decrypt aggregated gradients
    aggregated_grads, decryption_time = decrypt_gradients(aggregated_encrypted, original_shapes, fhe_context)

    # Collect metrics
    metrics = {
        'encryption_time': total_encryption_time,
        'decryption_time': decryption_time,
        'communication_bytes': total_communication_bytes,
    }

    return aggregated_grads, metrics


if __name__ == "__main__":
    # Test FedHE-Full
    print("Testing FedHE-Full method...")

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
    print("Running FedHE-Full round...")
    aggregated, metrics = fedhe_full_round(client_grads, fhe)

    print("\n--- Results ---")
    print(f"Encryption time: {metrics['encryption_time']:.4f} seconds")
    print(f"Decryption time: {metrics['decryption_time']:.4f} seconds")
    print(f"Communication: {metrics['communication_bytes'] / (1024*1024):.2f} MB")

    # Verify correctness (compare with plaintext averaging)
    print("\n--- Verification ---")
    for param_name in client_grads[0].keys():
        # Plaintext average
        plaintext_avg = np.mean([grads[param_name] for grads in client_grads], axis=0)

        # Compare with decrypted result
        max_error = np.max(np.abs(plaintext_avg - aggregated[param_name]))
        print(f"{param_name}: max error = {max_error:.10f}")
