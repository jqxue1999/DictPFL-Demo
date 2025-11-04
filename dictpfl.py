"""
DictPFL: Dictionary decomposition (DePE) + Pruning with Reactivation (PrME).
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from fhe_utils import FHEContext, get_ciphertext_size
from Pyfhel import PyCtxt


class DictPFLManager:
    """
    Manages dictionary decomposition for model parameters.

    DePE: Decomposes weight matrices W ≈ D × T
    - D: Dictionary (fixed after initialization)
    - T: Lookup table (trained and encrypted)
    """

    def __init__(self, rank: int = 8):
        """
        Initialize DictPFL manager.

        Args:
            rank: Rank for SVD decomposition
        """
        self.rank = rank
        self.dictionaries = {}  # param_name -> D
        self.lookup_tables = {}  # param_name -> T
        self.original_shapes = {}
        self.param_types = {}  # Track which params are decomposed

    def initialize_decomposition(self, params: Dict[str, np.ndarray]):
        """
        Initialize dictionary decomposition for model parameters.

        Args:
            params: Dictionary of model parameters
        """
        for param_name, param_array in params.items():
            self.original_shapes[param_name] = param_array.shape

            # Only decompose 2D weight matrices (not biases)
            if len(param_array.shape) == 2:
                # SVD decomposition: W ≈ U @ S @ Vt
                # We use truncated SVD: W ≈ U_r @ S_r @ Vt_r
                # Dictionary D = U_r @ sqrt(S_r)
                # Lookup table T = sqrt(S_r) @ Vt_r

                m, n = param_array.shape
                r = min(self.rank, min(m, n))

                try:
                    U, S, Vt = np.linalg.svd(param_array, full_matrices=False)

                    # Truncate to rank r
                    U_r = U[:, :r]
                    S_r = S[:r]
                    Vt_r = Vt[:r, :]

                    # Create dictionary and lookup table
                    sqrt_S = np.sqrt(S_r)
                    D = U_r @ np.diag(sqrt_S)
                    T = np.diag(sqrt_S) @ Vt_r

                    self.dictionaries[param_name] = D
                    self.lookup_tables[param_name] = T
                    self.param_types[param_name] = 'decomposed'

                except np.linalg.LinAlgError:
                    # Fallback: don't decompose this parameter
                    self.lookup_tables[param_name] = param_array.copy()
                    self.param_types[param_name] = 'full'
            else:
                # Bias terms - don't decompose
                self.lookup_tables[param_name] = param_array.copy()
                self.param_types[param_name] = 'full'

    def params_to_lookup_tables(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert full parameters to lookup table representation.

        Args:
            params: Dictionary of model parameters

        Returns:
            Dictionary of lookup tables
        """
        lookup_tables = {}

        for param_name, param_array in params.items():
            if param_name in self.param_types:
                if self.param_types[param_name] == 'decomposed':
                    # Project back to lookup table space
                    # W = D @ T, so T = D^+ @ W (pseudoinverse)
                    D = self.dictionaries[param_name]
                    D_pinv = np.linalg.pinv(D)
                    T = D_pinv @ param_array
                    lookup_tables[param_name] = T
                else:
                    lookup_tables[param_name] = param_array.copy()
            else:
                lookup_tables[param_name] = param_array.copy()

        return lookup_tables

    def lookup_tables_to_params(self, lookup_tables: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert lookup tables back to full parameters.

        Args:
            lookup_tables: Dictionary of lookup tables

        Returns:
            Dictionary of full model parameters
        """
        params = {}

        for param_name, T in lookup_tables.items():
            if param_name in self.param_types:
                if self.param_types[param_name] == 'decomposed':
                    # Reconstruct: W = D @ T
                    D = self.dictionaries[param_name]
                    W = D @ T
                    params[param_name] = W
                else:
                    params[param_name] = T.copy()
            else:
                params[param_name] = T.copy()

        return params

    def get_lookup_table_gradients(self, param_gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert parameter gradients to lookup table gradients.

        For decomposed params: grad_T = D^T @ grad_W
        For full params: grad_T = grad_W

        Args:
            param_gradients: Gradients w.r.t. full parameters

        Returns:
            Gradients w.r.t. lookup tables
        """
        lookup_table_grads = {}

        for param_name, grad_W in param_gradients.items():
            if param_name in self.param_types:
                if self.param_types[param_name] == 'decomposed':
                    # Chain rule: grad_T = D^T @ grad_W
                    D = self.dictionaries[param_name]
                    grad_T = D.T @ grad_W
                    lookup_table_grads[param_name] = grad_T
                else:
                    lookup_table_grads[param_name] = grad_W.copy()
            else:
                lookup_table_grads[param_name] = grad_W.copy()

        return lookup_table_grads


def prune_gradients(gradients: Dict[str, np.ndarray], prune_ratio: float = 0.5, beta: float = 0.2) -> Dict[str, np.ndarray]:
    """
    PrME: Prune lowest magnitude gradients with probabilistic reactivation.

    Args:
        gradients: Dictionary of gradients
        prune_ratio: Fraction of gradients to prune (0.0 to 1.0)
        beta: Reactivation probability for pruned gradients

    Returns:
        Pruned gradients
    """
    pruned_grads = {}

    for param_name, grad_array in gradients.items():
        flat_grad = grad_array.flatten()

        # Find threshold for pruning
        threshold_idx = int(len(flat_grad) * prune_ratio)
        if threshold_idx > 0:
            sorted_abs_grads = np.sort(np.abs(flat_grad))
            threshold = sorted_abs_grads[threshold_idx]

            # Create mask: keep gradients above threshold
            keep_mask = np.abs(flat_grad) >= threshold

            # Reactivation: randomly keep some pruned gradients
            pruned_mask = ~keep_mask
            if np.any(pruned_mask) and beta > 0:
                n_pruned = np.sum(pruned_mask)
                n_reactivate = int(n_pruned * beta)
                if n_reactivate > 0:
                    pruned_indices = np.where(pruned_mask)[0]
                    reactivate_indices = np.random.choice(pruned_indices, size=n_reactivate, replace=False)
                    keep_mask[reactivate_indices] = True

            # Apply mask
            pruned_flat = flat_grad * keep_mask
            pruned_grads[param_name] = pruned_flat.reshape(grad_array.shape)
        else:
            pruned_grads[param_name] = grad_array.copy()

    return pruned_grads


def encrypt_lookup_table_gradients(gradients: Dict[str, np.ndarray],
                                   fhe_context: FHEContext) -> Tuple[Dict[str, List[PyCtxt]], int, float]:
    """
    Encrypt lookup table gradients (including pruned/zero elements).

    Note: We encrypt the full lookup table for simplicity. The compression comes from
    dictionary decomposition (smaller lookup tables vs full parameters) and pruning
    reduces the magnitude of encrypted values.

    Args:
        gradients: Dictionary of lookup table gradients (may contain pruned/zero elements)
        fhe_context: FHE context

    Returns:
        Tuple of (encrypted_gradients, communication_bytes, encryption_time)
    """
    start_time = time.time()
    encrypted_grads = {}
    total_bytes = 0

    for param_name, grad_array in gradients.items():
        # Encrypt the full gradient array (simpler aggregation)
        encrypted_list = fhe_context.encrypt_gradients(grad_array)
        encrypted_grads[param_name] = encrypted_list

        for ctxt in encrypted_list:
            total_bytes += get_ciphertext_size(ctxt)

    encryption_time = time.time() - start_time

    return encrypted_grads, total_bytes, encryption_time


def aggregate_encrypted_lookup_tables(client_encrypted_grads: List[Dict[str, List[PyCtxt]]],
                                      n_clients: int) -> Dict[str, List[PyCtxt]]:
    """
    Homomorphically aggregate encrypted lookup table gradients.

    Args:
        client_encrypted_grads: List of encrypted gradients from each client
        n_clients: Number of clients

    Returns:
        Aggregated encrypted gradients
    """
    if not client_encrypted_grads:
        return {}

    aggregated = {}
    param_names = client_encrypted_grads[0].keys()

    for param_name in param_names:
        n_ctxts = len(client_encrypted_grads[0][param_name])
        aggregated[param_name] = []

        for ctxt_idx in range(n_ctxts):
            ctxts_to_add = [
                client_grads[param_name][ctxt_idx]
                for client_grads in client_encrypted_grads
            ]

            aggregated_ctxt = ctxts_to_add[0].copy()
            for ctxt in ctxts_to_add[1:]:
                aggregated_ctxt += ctxt

            # Average
            aggregated_ctxt *= (1.0 / n_clients)

            aggregated[param_name].append(aggregated_ctxt)

    return aggregated


def decrypt_lookup_table_gradients(encrypted_grads: Dict[str, List[PyCtxt]],
                                   original_shapes: Dict[str, tuple],
                                   fhe_context: FHEContext) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Decrypt aggregated lookup table gradients.

    Args:
        encrypted_grads: Encrypted gradients
        original_shapes: Original shapes
        fhe_context: FHE context

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


def dictpfl_round(client_gradients: List[Dict[str, np.ndarray]],
                 fhe_context: FHEContext,
                 dictpfl_manager: DictPFLManager,
                 prune_ratio: float = 0.5,
                 beta: float = 0.2) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Execute one round of DictPFL (DePE + PrME).

    Args:
        client_gradients: List of gradient dictionaries from each client
        fhe_context: FHE context
        dictpfl_manager: DictPFL manager for decomposition
        prune_ratio: Pruning ratio for PrME
        beta: Reactivation probability for PrME

    Returns:
        Tuple of (aggregated_gradients, metrics_dict)
    """
    n_clients = len(client_gradients)
    total_encryption_time = 0.0
    total_communication_bytes = 0

    # Step 1: Convert gradients to lookup table space and apply pruning
    client_lt_grads_encrypted = []
    lt_shapes = {}

    for client_grads in client_gradients:
        # Convert to lookup table gradients
        lt_grads = dictpfl_manager.get_lookup_table_gradients(client_grads)

        # Apply PrME (pruning with reactivation)
        pruned_lt_grads = prune_gradients(lt_grads, prune_ratio=prune_ratio, beta=beta)

        # Encrypt lookup table gradients
        encrypted_grads, comm_bytes, enc_time = encrypt_lookup_table_gradients(pruned_lt_grads, fhe_context)

        client_lt_grads_encrypted.append(encrypted_grads)
        total_encryption_time += enc_time
        total_communication_bytes += comm_bytes

        # Store shapes
        if not lt_shapes:
            for param_name, grad_array in pruned_lt_grads.items():
                lt_shapes[param_name] = grad_array.shape

    # Step 2: Server aggregates encrypted lookup table gradients
    aggregated_encrypted = aggregate_encrypted_lookup_tables(client_lt_grads_encrypted, n_clients)

    # Communication: server sends back aggregated gradients
    total_communication_bytes += total_communication_bytes // n_clients

    # Step 3: Decrypt aggregated lookup table gradients
    aggregated_lt_grads, decryption_time = decrypt_lookup_table_gradients(
        aggregated_encrypted, lt_shapes, fhe_context
    )

    # Step 4: Convert lookup table gradients back to parameter space
    aggregated_grads = {}
    for param_name, lt_grad in aggregated_lt_grads.items():
        if param_name in dictpfl_manager.param_types:
            if dictpfl_manager.param_types[param_name] == 'decomposed':
                # grad_W = D @ grad_T
                D = dictpfl_manager.dictionaries[param_name]
                grad_W = D @ lt_grad
                aggregated_grads[param_name] = grad_W
            else:
                aggregated_grads[param_name] = lt_grad
        else:
            aggregated_grads[param_name] = lt_grad

    # Collect metrics
    metrics = {
        'encryption_time': total_encryption_time,
        'decryption_time': decryption_time,
        'communication_bytes': total_communication_bytes,
    }

    return aggregated_grads, metrics


if __name__ == "__main__":
    # Test DictPFL
    print("Testing DictPFL method...")

    # Create FHE context
    print("Initializing FHE context...")
    fhe = FHEContext(n=8192, scale=30)

    # Create DictPFL manager
    print("Initializing DictPFL manager...")
    dictpfl_mgr = DictPFLManager(rank=8)

    # Initialize with model parameters
    init_params = {
        'fc1.weight': np.random.randn(32, 2) * 0.1,
        'fc1.bias': np.random.randn(32) * 0.1,
        'fc2.weight': np.random.randn(2, 32) * 0.1,
        'fc2.bias': np.random.randn(2) * 0.1,
    }
    dictpfl_mgr.initialize_decomposition(init_params)

    print("Decomposition complete:")
    for param_name in init_params.keys():
        if param_name in dictpfl_mgr.dictionaries:
            print(f"  {param_name}: D shape = {dictpfl_mgr.dictionaries[param_name].shape}, "
                  f"T shape = {dictpfl_mgr.lookup_tables[param_name].shape}")

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
    print("\nRunning DictPFL round...")
    aggregated, metrics = dictpfl_round(client_grads, fhe, dictpfl_mgr, prune_ratio=0.5, beta=0.2)

    print("\n--- Results ---")
    print(f"Encryption time: {metrics['encryption_time']:.4f} seconds")
    print(f"Decryption time: {metrics['decryption_time']:.4f} seconds")
    print(f"Communication: {metrics['communication_bytes'] / (1024*1024):.2f} MB")

    print("\n--- Aggregated gradient shapes ---")
    for param_name, grad_array in aggregated.items():
        print(f"  {param_name}: {grad_array.shape}")
