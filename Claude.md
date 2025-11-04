# 🧠 DictPFL Federated Learning Demo (with FHE Encryption)

**Author:** Jiaqi Xue  
**Objective:** Build an interactive web demo for the paper *“DictPFL: Efficient and Private Federated Learning on Encrypted Gradients”*  
**Tech Stack:** Python · PyTorch · Pyfhel (CKKS) · Gradio · NumPy · scikit-learn · matplotlib  

---

## 🎯 Project Overview

The goal is to develop a **lightweight interactive demo** that demonstrates the **DictPFL framework** versus two baselines:
1. **FedHE-Full** — all gradients encrypted
2. **FedML-HE** — encrypt top-10% gradients (partial encryption)
3. **DictPFL** — DePE (Dictionary Decomposition) + PrME (Pruning with Reactivation)

The demo uses **real FHE operations** (encryption, decryption, ciphertext addition) via **Pyfhel (CKKS)**, and trains a small model (2-layer MLP) on a simple dataset (`make_moons` or MNIST subset).

Target runtime: **< 60 seconds total**, **< 1 second per round** (toy-scale).

---

## 🧩 System Architecture

### 🧠 Core Components

| File | Description |
|------|--------------|
| `demo.py` | Main Gradio app orchestrating training, encryption, aggregation, and visualization |
| `dataset.py` | Generates and partitions small dataset among clients |
| `model.py` | Defines small PyTorch MLP |
| `fhe_utils.py` | Wrapper for Pyfhel-based FHE: encryption, decryption, ciphertext addition |
| `fedhe_full.py` | Baseline method: encrypt all gradients |
| `fedml_he.py` | Baseline method: encrypt top-10% gradients |
| `dictpfl.py` | DictPFL implementation (DePE + PrME) |
| `metrics.py` | Tracks communication, time, and accuracy |
| `plots.py` | Live charts for training metrics |
| `README.md` | Instructions for running and extending the demo |

---

## ⚙️ Functional Requirements

### 1. Federated Learning Simulation
- 5 clients (default), each with local dataset partition
- Global model synchronized each round
- Server aggregates gradients homomorphically

### 2. Methods
| Method | Encryption Strategy | Description |
|---------|----------------------|--------------|
| FedHE-Full | Encrypt all gradients | Full privacy, slowest |
| FedML-HE | Encrypt top 10% by magnitude | Partial privacy, faster |
| DictPFL | Encrypt lookup table gradients only (DePE + PrME) | Full privacy, efficient |

### 3. DictPFL Modules
- **DePE:**  
  - Factorize model weights via truncated SVD (rank `r`).  
  - Keep dictionary `D` fixed; encrypt & train lookup table `T`.  
- **PrME:**  
  - Prune lowest `s%` gradients.  
  - Reactivate with probability β=0.2.  

### 4. FHE Operations (via Pyfhel)
- Context: CKKS scheme (`n=8192, scale=2^30`)
- Only these operations are used:
  - `encryptFrac()`
  - `decryptFrac()`
  - `__add__` for ciphertext aggregation
- No ciphertext multiplications or bootstrapping

---

## 🧮 Computation Flow

```text
Clients (5)
 ├─ Local training → gradients
 ├─ Encrypt gradients (partial/full)
 ├─ Send to server
 ├─ Receive the aggregated encrypted gradients from the server
 └─ Decrpyt and update model
Server
 ├─ Homomorphic sum (ciphertext addition)
 └─ Send to Client
