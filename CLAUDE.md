# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **DictPFL: Efficient and Private Federated Learning on Encrypted Gradients**, demonstrating dictionary decomposition (DePE) and pruning with reactivation (PrME) techniques for efficient homomorphic encryption in federated learning.

The project implements an interactive Gradio demo comparing three methods:
1. **FedHE-Full** - Full gradient encryption (baseline)
2. **FedML-HE** - Partial encryption (top 10% gradients)
3. **DictPFL** - Dictionary decomposition with pruned encrypted gradients

Target: < 60 seconds total training time, < 1 second per round on toy-scale datasets.

## Development Setup

**Installation:**
```bash
pip install torch numpy scikit-learn matplotlib gradio Pyfhel
```

**Running the demo:**
```bash
python demo.py
```

**Running tests (when implemented):**
```bash
python -m pytest tests/
python -m pytest tests/test_fhe_utils.py -v  # single test file
```

## Architecture

### Core Components Structure

**Federated Learning Pipeline:**
- `demo.py` - Main Gradio application entry point, orchestrates the entire FL workflow
- `dataset.py` - Dataset generation (make_moons or MNIST subset) and non-IID partitioning across clients
- `model.py` - 2-layer MLP PyTorch model definition

**Encryption Layer (Pyfhel/CKKS):**
- `fhe_utils.py` - FHE wrapper using Pyfhel library (CKKS scheme: n=8192, scale=2^30)
  - Only operations: encryptFrac(), decryptFrac(), ciphertext addition
  - No multiplication or bootstrapping required

**Method Implementations:**
- `fedhe_full.py` - Baseline: encrypt all gradient elements
- `fedml_he.py` - Baseline: encrypt top 10% gradients by magnitude
- `dictpfl.py` - DictPFL implementation:
  - DePE: SVD factorization (W ≈ D × T), encrypt only lookup table T gradients
  - PrME: Gradient pruning with probabilistic reactivation (β=0.2)

**Monitoring & Visualization:**
- `metrics.py` - Track communication cost, encryption time, training time, accuracy
- `plots.py` - Real-time training metric visualization

### Key Architectural Patterns

**Simulation Flow:**
```
Client-side (5 clients):
1. Local training → compute gradients
2. Method-specific gradient selection/transformation
3. Encrypt selected gradients
4. Send encrypted gradients to server
5. Receive aggregated encrypted gradients
6. Decrypt and apply to local model

Server-side:
1. Receive encrypted gradients from all clients
2. Homomorphic aggregation (ciphertext addition only)
3. Broadcast aggregated encrypted gradients back to clients
```

**DictPFL Specific Logic:**
- Dictionary D is fixed after initial SVD decomposition (rank r)
- Only lookup table T is trained and encrypted
- Gradient pruning: zero out lowest s% of gradient magnitudes
- Reactivation: restore pruned gradients with probability β=0.2

**FHE Context Management:**
- Single CKKS context shared across all clients (simplified for demo)
- In production: each client would generate keys, send public key to server
- Context parameters: poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60], scale=2^30

## Important Implementation Notes

**FHE Operations:**
- Use Pyfhel's `encryptFrac()` for floating-point encoding
- All ciphertext operations are additions (homomorphic sum)
- No ciphertext rotations or multiplications needed
- Batch encryption: pack multiple gradients into single ciphertext when possible

**Dataset Partitioning:**
- Non-IID split: use label skew or quantity skew
- Default: 5 clients, each gets ~20% of data with distribution imbalance

**Performance Targets:**
- Total runtime: < 60 seconds for demo
- Per-round time: < 1 second
- Use small models (hundreds of parameters, not millions)
- Dataset: 1000-5000 samples total

**Privacy Guarantees:**
- FedHE-Full: Full gradient-level privacy, highest overhead
- FedML-HE: Partial privacy (90% plaintext exposure)
- DictPFL: Full privacy with reduced communication (encrypt smaller lookup table)

## Testing Strategy

When implementing tests:
- Unit tests for FHE operations (encrypt/decrypt correctness, homomorphic addition)
- Integration tests for each method (end-to-end gradient aggregation)
- Performance benchmarks (time per round, communication bytes)
- Convergence tests (verify accuracy targets are met)

## Reference Paper

The research paper is available at `2510.21086v1.pdf` in the repository root. Refer to it for:
- Mathematical formulations of DePE and PrME
- Experimental setup details
- Privacy and efficiency analysis
