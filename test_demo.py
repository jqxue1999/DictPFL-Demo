"""
Quick test to verify the demo works without launching the UI.
"""

import numpy as np
import torch

# Test imports
print("Testing imports...")
from dataset import get_federated_data, generate_dataset
from model import SimpleMLP, get_model_params, set_model_params, train_one_epoch, evaluate_model
from fhe_utils import FHEContext
from metrics import MetricsTracker
from fedhe_full import fedhe_full_round
from fedml_he import fedml_he_round
from dictpfl import DictPFLManager, dictpfl_round

print("✓ All imports successful")

# Test data generation
print("\nTesting data generation...")
np.random.seed(42)
torch.manual_seed(42)

client_datasets = get_federated_data(
    dataset_name='moons',
    n_clients=3,
    n_samples=300,
    non_iid=True,
    alpha=0.5,
    seed=42
)

X_test, y_test = generate_dataset('moons', n_samples=100, seed=43)
print(f"✓ Created {len(client_datasets)} clients")
for i, (X, y) in enumerate(client_datasets):
    print(f"  Client {i}: {len(X)} samples")

# Test FHE context
print("\nTesting FHE context...")
fhe = FHEContext(n=8192, scale=30)
print("✓ FHE context initialized")

# Test DictPFL manager
print("\nTesting DictPFL manager...")
model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
dictpfl_mgr = DictPFLManager(rank=8)
init_params = get_model_params(model)
dictpfl_mgr.initialize_decomposition(init_params)
print("✓ DictPFL manager initialized")

# Test one training round for each method
print("\nTesting training round...")
global_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
global_params = get_model_params(global_model)

# Collect gradients
client_gradients = []
for X_client, y_client in client_datasets:
    client_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
    set_model_params(client_model, global_params)

    loss = train_one_epoch(client_model, X_client, y_client, lr=0.01, batch_size=32)

    updated_params = get_model_params(client_model)
    gradients = {name: updated_params[name] - global_params[name]
                for name in global_params.keys()}
    client_gradients.append(gradients)

print(f"✓ Collected gradients from {len(client_gradients)} clients")

# Test FedHE-Full
print("\nTesting FedHE-Full...")
agg_grads, metrics = fedhe_full_round(client_gradients, fhe)
print(f"✓ FedHE-Full: enc={metrics['encryption_time']:.3f}s, comm={metrics['communication_bytes']/1024:.2f}KB")

# Test FedML-HE
print("\nTesting FedML-HE...")
agg_grads, metrics = fedml_he_round(client_gradients, fhe, k_percent=0.1)
print(f"✓ FedML-HE: enc={metrics['encryption_time']:.3f}s, comm={metrics['communication_bytes']/1024:.2f}KB")

# Test DictPFL
print("\nTesting DictPFL...")
agg_grads, metrics = dictpfl_round(client_gradients, fhe, dictpfl_mgr, prune_ratio=0.5, beta=0.2)
print(f"✓ DictPFL: enc={metrics['encryption_time']:.3f}s, comm={metrics['communication_bytes']/1024:.2f}KB")

# Update model and evaluate
for param_name in global_params.keys():
    global_params[param_name] = global_params[param_name] + agg_grads[param_name]
set_model_params(global_model, global_params)

accuracy = evaluate_model(global_model, X_test, y_test)
print(f"\nModel accuracy after 1 round: {accuracy:.4f}")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED - Demo is ready to run!")
print("="*50)
print("\nTo launch the demo, run: python demo.py")
