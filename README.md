# DictPFL: Efficient and Private Federated Learning Demo

Interactive demonstration of **DictPFL** (Dictionary-based Private Federated Learning) comparing three homomorphic encryption approaches for federated learning.

## Overview

This project implements and compares three federated learning methods with FHE (Fully Homomorphic Encryption):

1. **FedHE-Full**: Encrypt all gradients (full privacy, highest overhead)
2. **FedML-HE**: Encrypt top-10% gradients by magnitude (partial privacy, reduced overhead)
3. **DictPFL**: Dictionary decomposition (DePE) + Pruning with Reactivation (PrME) (full privacy, most efficient)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch
- Pyfhel (for FHE operations)
- Gradio (for web interface)
- NumPy, scikit-learn, matplotlib

## Usage

### Option 1: Command-Line Experiments

Run the comparison experiment without GUI:

```bash
python experiment.py
```

This will:
- Generate a non-IID federated dataset (make_moons)
- Train all three methods for 10 rounds
- Compare their performance, efficiency, and communication costs
- Generate comparison plots (saved as PNG files)

### Option 2: Interactive Demo

Launch the Gradio web interface:

```bash
python demo.py
```

Then open your browser to the provided URL (typically http://localhost:7860).

The demo provides three tabs:
1. **System Setup**: Initialize the federated learning environment
2. **Single Method Training**: Train and evaluate one method at a time
3. **Method Comparison**: Run all three methods and compare results

## Project Structure

```
DictPFL/
├── dataset.py          # Data generation and non-IID partitioning
├── model.py            # Simple 2-layer MLP model
├── fhe_utils.py        # Pyfhel wrapper for CKKS encryption
├── metrics.py          # Performance tracking and comparison
├── fedhe_full.py       # FedHE-Full baseline (encrypt all)
├── fedml_he.py         # FedML-HE baseline (encrypt top-k%)
├── dictpfl.py          # DictPFL implementation (DePE + PrME)
├── plots.py            # Visualization utilities
├── experiment.py       # Command-line comparison script
├── demo.py             # Gradio web interface
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── CLAUDE.md           # Developer guidance
```

## Key Components

### DictPFL Algorithm

**DePE (Dictionary Decomposition):**
- Factorize weight matrices: W ≈ D × T
- Dictionary D is fixed after SVD initialization
- Only lookup table T is trained and encrypted
- Reduces communication overhead significantly

**PrME (Pruning with Reactivation):**
- Prune lowest s% of gradients by magnitude
- Reactivate pruned gradients with probability β=0.2
- Further reduces communication while maintaining accuracy

### FHE Configuration

- **Scheme**: CKKS (approximate homomorphic encryption for floats)
- **Parameters**: n=8192, scale=2^30
- **Operations**: Encryption, decryption, ciphertext addition only
- **Library**: Pyfhel

## Performance Targets

This is a toy-scale demo designed for:
- **Total runtime**: < 60 seconds
- **Per-round time**: < 1 second
- **Dataset**: 2000 samples, 5 clients
- **Model**: 2-layer MLP (~300 parameters)

For production-scale systems, scale up the model size, dataset, and FHE parameters accordingly.

## Testing Individual Components

Each module can be tested independently:

```bash
# Test dataset generation
python dataset.py

# Test model training
python model.py

# Test FHE operations
python fhe_utils.py

# Test FedHE-Full method
python fedhe_full.py

# Test FedML-HE method
python fedml_he.py

# Test DictPFL method
python dictpfl.py

# Test metrics tracking
python metrics.py

# Test plotting
python plots.py
```

## Expected Results

**Typical performance comparison (10 rounds, 5 clients, 2000 samples):**

| Method      | Total Time | Communication | Final Accuracy | Privacy Level |
|-------------|-----------|---------------|----------------|---------------|
| FedHE-Full  | ~20-30s   | ~50-100 MB    | 0.85-0.90      | Full (100%)   |
| FedML-HE    | ~15-20s   | ~25-50 MB     | 0.85-0.90      | Partial (10%) |
| DictPFL     | ~15-25s   | ~30-70 MB     | 0.85-0.90      | Full (100%)   |

**Key Insights:**
- All three methods achieve **similar accuracy** when using the same initial weights
- **FedML-HE** is fastest but only encrypts 10% of gradients (90% exposed in plaintext!)
- **DictPFL** provides full privacy with reduced communication via dictionary decomposition
- For this toy model (82 parameters → 54 lookup elements), compression is modest (1.52x)
- **DictPFL shines with larger models** where SVD compression is more dramatic (e.g., 512×128 → rank-64 = 2x)

*Actual times depend on hardware and FHE implementation.*

## Research Paper

This implementation is based on the paper:
**"DictPFL: Efficient and Private Federated Learning on Encrypted Gradients"**

See `2510.21086v1.pdf` for detailed methodology and theoretical analysis.

## Notes

- FHE operations are computationally expensive; expect significant runtime
- For faster testing, reduce `n_rounds` or `n_samples`
- The demo uses a simplified threat model (shared FHE context)
- In production, each client would have their own key pair

## Troubleshooting

**Pyfhel installation issues:**
```bash
# On Linux, you may need to install build dependencies
sudo apt-get install python3-dev
pip install --upgrade pip setuptools wheel
pip install Pyfhel
```

**Out of memory:**
- Reduce `n_samples` or `n_clients`
- Use smaller FHE parameters (e.g., n=4096)

**Slow performance:**
- This is expected with FHE operations
- Consider running on a machine with more CPU cores
- Reduce the number of rounds for testing

## License

This is a research implementation for educational purposes.
