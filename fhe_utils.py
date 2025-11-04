"""
FHE utilities using Pyfhel library with CKKS scheme.
"""

from Pyfhel import Pyfhel, PyCtxt
import numpy as np
from typing import List, Union
import time


class FHEContext:
    """
    Wrapper for Pyfhel CKKS context for encrypting and operating on floating-point data.
    """

    def __init__(self, n: int = 8192, scale: int = 30, qi_sizes: List[int] = None):
        """
        Initialize FHE context with CKKS scheme.

        Args:
            n: Polynomial modulus degree (power of 2, typically 8192 or 16384)
            scale: Scale bits for encoding (typically 30-40)
            qi_sizes: Coefficient modulus bit sizes
        """
        self.HE = Pyfhel()
        self.n = n
        self.scale = scale

        if qi_sizes is None:
            # Default: [60, 40, 40, 60] for good precision
            qi_sizes = [60, 40, 40, 60]

        # Generate context
        self.HE.contextGen(scheme='ckks', n=n, scale=2**scale, qi_sizes=qi_sizes)

        # Generate keys
        self.HE.keyGen()

        # For demo purposes, we'll use rotation keys if needed
        # self.HE.rotateKeyGen()

    def encrypt(self, value: Union[float, np.ndarray]) -> PyCtxt:
        """
        Encrypt a float or array of floats.

        Args:
            value: Single float or numpy array

        Returns:
            Encrypted ciphertext
        """
        if isinstance(value, (int, float)):
            value = np.array([value], dtype=np.float64)
        elif isinstance(value, np.ndarray):
            value = value.flatten().astype(np.float64)

        # Encode and encrypt
        ctxt = self.HE.encryptFrac(value)
        return ctxt

    def decrypt(self, ctxt: PyCtxt) -> np.ndarray:
        """
        Decrypt a ciphertext to numpy array.

        Args:
            ctxt: Encrypted ciphertext

        Returns:
            Decrypted values as numpy array
        """
        result = self.HE.decryptFrac(ctxt)
        return np.array(result)

    def add_ciphertexts(self, ctxt_list: List[PyCtxt]) -> PyCtxt:
        """
        Homomorphically add multiple ciphertexts.

        Args:
            ctxt_list: List of ciphertexts to add

        Returns:
            Sum of all ciphertexts
        """
        if not ctxt_list:
            raise ValueError("Cannot add empty list of ciphertexts")

        result = ctxt_list[0].copy()
        for ctxt in ctxt_list[1:]:
            result += ctxt

        return result

    def encrypt_gradients(self, gradients: np.ndarray) -> List[PyCtxt]:
        """
        Encrypt gradient array, potentially batching for efficiency.

        Args:
            gradients: Gradient array to encrypt

        Returns:
            List of encrypted ciphertexts
        """
        flat_grads = gradients.flatten()

        # For simplicity in this demo, encrypt in small batches
        # In production, you'd batch more efficiently based on slot count
        batch_size = min(100, len(flat_grads))

        encrypted_grads = []
        for i in range(0, len(flat_grads), batch_size):
            batch = flat_grads[i:i + batch_size]
            ctxt = self.encrypt(batch)
            encrypted_grads.append(ctxt)

        return encrypted_grads

    def decrypt_gradients(self, encrypted_grads: List[PyCtxt], original_shape: tuple) -> np.ndarray:
        """
        Decrypt list of ciphertexts back to gradient array.

        Args:
            encrypted_grads: List of encrypted ciphertexts
            original_shape: Original shape to reshape to

        Returns:
            Decrypted gradient array
        """
        decrypted_parts = []
        for ctxt in encrypted_grads:
            decrypted_parts.append(self.decrypt(ctxt))

        # Concatenate all parts
        flat_result = np.concatenate(decrypted_parts)

        # Reshape to original shape
        total_elements = np.prod(original_shape)
        result = flat_result[:total_elements].reshape(original_shape)

        return result


def measure_encryption_time(fhe_context: FHEContext, data: np.ndarray) -> float:
    """
    Measure time to encrypt data.

    Args:
        fhe_context: FHE context
        data: Data to encrypt

    Returns:
        Encryption time in seconds
    """
    start_time = time.time()
    _ = fhe_context.encrypt(data)
    end_time = time.time()
    return end_time - start_time


def measure_decryption_time(fhe_context: FHEContext, ctxt: PyCtxt) -> float:
    """
    Measure time to decrypt ciphertext.

    Args:
        fhe_context: FHE context
        ctxt: Ciphertext to decrypt

    Returns:
        Decryption time in seconds
    """
    start_time = time.time()
    _ = fhe_context.decrypt(ctxt)
    end_time = time.time()
    return end_time - start_time


def get_ciphertext_size(ctxt: PyCtxt) -> int:
    """
    Estimate ciphertext size in bytes.

    Args:
        ctxt: Ciphertext

    Returns:
        Approximate size in bytes
    """
    # Pyfhel ciphertexts size depends on parameters
    # For CKKS with n=8192, each ciphertext is roughly 100-200 KB
    # This is a rough estimate for demo purposes
    return len(ctxt.to_bytes())


if __name__ == "__main__":
    # Test FHE operations
    print("Initializing FHE context...")
    fhe = FHEContext(n=8192, scale=30)
    print("FHE context initialized successfully!")

    # Test encryption/decryption
    print("\n--- Testing encryption/decryption ---")
    value = 3.14159
    ctxt = fhe.encrypt(value)
    decrypted = fhe.decrypt(ctxt)
    print(f"Original: {value}")
    print(f"Decrypted: {decrypted[0]:.6f}")
    print(f"Error: {abs(value - decrypted[0]):.10f}")

    # Test array encryption
    print("\n--- Testing array encryption ---")
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ctxt_arr = fhe.encrypt(arr)
    decrypted_arr = fhe.decrypt(ctxt_arr)
    print(f"Original: {arr}")
    print(f"Decrypted: {decrypted_arr[:len(arr)]}")

    # Test homomorphic addition
    print("\n--- Testing homomorphic addition ---")
    val1 = np.array([1.0, 2.0])
    val2 = np.array([3.0, 4.0])
    ctxt1 = fhe.encrypt(val1)
    ctxt2 = fhe.encrypt(val2)
    ctxt_sum = fhe.add_ciphertexts([ctxt1, ctxt2])
    decrypted_sum = fhe.decrypt(ctxt_sum)
    print(f"val1: {val1}")
    print(f"val2: {val2}")
    print(f"Expected sum: {val1 + val2}")
    print(f"Decrypted sum: {decrypted_sum[:len(val1)]}")

    # Test gradient encryption
    print("\n--- Testing gradient encryption ---")
    grads = np.random.randn(50)
    encrypted_grads = fhe.encrypt_gradients(grads)
    decrypted_grads = fhe.decrypt_gradients(encrypted_grads, grads.shape)
    print(f"Gradient shape: {grads.shape}")
    print(f"Number of ciphertexts: {len(encrypted_grads)}")
    print(f"Max error: {np.max(np.abs(grads - decrypted_grads)):.10f}")

    # Measure times
    print("\n--- Performance metrics ---")
    test_data = np.random.randn(100)
    enc_time = measure_encryption_time(fhe, test_data)
    print(f"Encryption time (100 values): {enc_time*1000:.2f} ms")

    ctxt_test = fhe.encrypt(test_data)
    dec_time = measure_decryption_time(fhe, ctxt_test)
    print(f"Decryption time: {dec_time*1000:.2f} ms")

    ctxt_size = get_ciphertext_size(ctxt_test)
    print(f"Ciphertext size: {ctxt_size / 1024:.2f} KB")
