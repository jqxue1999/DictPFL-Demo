"""
Gradio demo for DictPFL: Interactive federated learning comparison.
"""

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from dataset import get_federated_data, generate_dataset
from model import SimpleMLP, train_one_epoch, evaluate_model, get_model_gradients, set_model_params, get_model_params
from fhe_utils import FHEContext
from metrics import MetricsTracker, ComparisonMetrics
from plots import plot_training_metrics, plot_efficiency_metrics, plot_comparison_bars, plot_encryption_breakdown

from fedhe_full import fedhe_full_round
from fedml_he import fedml_he_round
from dictpfl import DictPFLManager, dictpfl_round


# Global state
global_state = {
    'fhe_context': None,
    'client_datasets': None,
    'test_data': None,
    'dictpfl_manager': None,
}


def initialize_system(n_clients: int, n_samples: int, seed: int) -> str:
    """
    Initialize the federated learning system.

    Args:
        n_clients: Number of clients
        n_samples: Total number of samples
        seed: Random seed

    Returns:
        Status message
    """
    try:
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate and partition data
        client_datasets = get_federated_data(
            dataset_name='moons',
            n_clients=n_clients,
            n_samples=n_samples,
            non_iid=True,
            alpha=0.5,
            seed=seed
        )

        # Create test dataset
        X_test, y_test = generate_dataset('moons', n_samples=500, seed=seed+1)
        test_data = (X_test, y_test)

        # Initialize FHE context
        fhe = FHEContext(n=8192, scale=30)

        # Initialize DictPFL manager
        dictpfl_mgr = DictPFLManager(rank=8)
        init_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
        init_params = get_model_params(init_model)
        dictpfl_mgr.initialize_decomposition(init_params)

        # Store in global state
        global_state['fhe_context'] = fhe
        global_state['client_datasets'] = client_datasets
        global_state['test_data'] = test_data
        global_state['dictpfl_manager'] = dictpfl_mgr

        status = f"✓ System initialized successfully!\n\n"
        status += f"Clients: {n_clients}\n"
        status += f"Total samples: {n_samples}\n"
        for i, (X_client, y_client) in enumerate(client_datasets):
            status += f"  Client {i}: {len(X_client)} samples\n"
        status += f"\nFHE context ready (CKKS scheme, n=8192)\n"
        status += f"DictPFL decomposition initialized (rank=8)\n"

        return status

    except Exception as e:
        return f"Error during initialization: {str(e)}"


def run_single_method(method_name: str, n_rounds: int, lr: float,
                     k_percent: float, prune_ratio: float, beta: float,
                     progress=gr.Progress()) -> Tuple:
    """
    Run federated training for a single method.

    Args:
        method_name: Method to run
        n_rounds: Number of rounds
        lr: Learning rate
        k_percent: Percentage for FedML-HE
        prune_ratio: Pruning ratio for DictPFL
        beta: Reactivation probability for DictPFL
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_text, training_plot, efficiency_plot)
    """
    if global_state['fhe_context'] is None:
        return "Error: Please initialize the system first!", None, None

    try:
        fhe = global_state['fhe_context']
        client_datasets = global_state['client_datasets']
        test_data = global_state['test_data']
        X_test, y_test = test_data
        n_clients = len(client_datasets)

        # Initialize global model
        global_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)

        # Initialize metrics tracker
        tracker = MetricsTracker()

        # Initialize client models
        client_models = [SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2) for _ in range(n_clients)]

        # Get DictPFL manager (re-initialize for fresh run)
        if method_name == 'DictPFL':
            dictpfl_mgr = DictPFLManager(rank=8)
            init_params = get_model_params(global_model)
            dictpfl_mgr.initialize_decomposition(init_params)
        else:
            dictpfl_mgr = None

        status = f"Running {method_name}...\n\n"

        # Training loop
        for round_num in progress.tqdm(range(1, n_rounds + 1), desc=f"Training {method_name}"):
            tracker.start_round(round_num)

            # Broadcast global model to clients
            global_params = get_model_params(global_model)
            for client_model in client_models:
                set_model_params(client_model, global_params)

            # Local training and gradient computation
            client_gradients = []
            total_loss = 0.0

            for client_idx, (X_client, y_client) in enumerate(client_datasets):
                loss = train_one_epoch(client_models[client_idx], X_client, y_client, lr=lr, batch_size=32)
                total_loss += loss

                updated_params = get_model_params(client_models[client_idx])
                gradients = {}
                for param_name in global_params.keys():
                    gradients[param_name] = updated_params[param_name] - global_params[param_name]

                client_gradients.append(gradients)

            avg_loss = total_loss / n_clients

            # Aggregate gradients
            if method_name == 'FedHE-Full':
                aggregated_grads, method_metrics = fedhe_full_round(client_gradients, fhe)
            elif method_name == 'FedML-HE':
                aggregated_grads, method_metrics = fedml_he_round(client_gradients, fhe, k_percent=k_percent)
            elif method_name == 'DictPFL':
                aggregated_grads, method_metrics = dictpfl_round(
                    client_gradients, fhe, dictpfl_mgr, prune_ratio=prune_ratio, beta=beta
                )
            else:
                return f"Error: Unknown method {method_name}", None, None

            # Update tracker
            tracker.add_encryption_time(method_metrics['encryption_time'])
            tracker.add_decryption_time(method_metrics['decryption_time'])
            tracker.add_communication_bytes(method_metrics['communication_bytes'])

            # Update global model
            updated_global_params = {}
            for param_name in global_params.keys():
                updated_global_params[param_name] = global_params[param_name] + aggregated_grads[param_name]
            set_model_params(global_model, updated_global_params)

            # Evaluate
            accuracy = evaluate_model(global_model, X_test, y_test)
            tracker.set_accuracy(accuracy)
            tracker.set_loss(avg_loss)

            tracker.end_round()

            status += f"Round {round_num}: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}\n"

        # Generate summary
        summary = tracker.get_summary()
        status += f"\n{'='*50}\n"
        status += f"RESULTS for {method_name}\n"
        status += f"{'='*50}\n"
        status += f"Total Time: {summary['total_time']:.2f}s\n"
        status += f"Avg Round Time: {summary['avg_round_time']:.4f}s\n"
        status += f"Total Encryption: {summary['total_encryption_time']:.2f}s\n"
        status += f"Total Decryption: {summary['total_decryption_time']:.2f}s\n"
        status += f"Total Communication: {summary['total_communication_mb']:.2f} MB\n"
        status += f"Final Accuracy: {summary['final_accuracy']:.4f}\n"

        # Generate plots
        round_data = tracker.get_round_data()
        methods_data = {method_name: round_data}

        fig1 = plot_training_metrics(methods_data)
        fig2 = plot_efficiency_metrics(methods_data)

        return status, fig1, fig2

    except Exception as e:
        return f"Error during training: {str(e)}", None, None


def run_comparison(n_rounds: int, lr: float, progress=gr.Progress()) -> Tuple:
    """
    Run comparison of all three methods.

    Args:
        n_rounds: Number of rounds
        lr: Learning rate
        progress: Gradio progress tracker

    Returns:
        Tuple of plots and status
    """
    if global_state['fhe_context'] is None:
        return "Error: Please initialize the system first!", None, None, None, None

    try:
        results = {}
        methods = ['FedHE-Full', 'FedML-HE', 'DictPFL']

        status = "Running comparison of all methods...\n\n"

        for method_idx, method_name in enumerate(methods):
            fhe = global_state['fhe_context']
            client_datasets = global_state['client_datasets']
            test_data = global_state['test_data']
            X_test, y_test = test_data
            n_clients = len(client_datasets)

            progress((method_idx / len(methods), method_idx + 1, len(methods)),
                    desc=f"Running {method_name}")

            # Initialize models
            global_model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2)
            client_models = [SimpleMLP(input_dim=2, hidden_dim=32, output_dim=2) for _ in range(n_clients)]
            tracker = MetricsTracker()

            # Initialize DictPFL manager if needed
            if method_name == 'DictPFL':
                dictpfl_mgr = DictPFLManager(rank=8)
                init_params = get_model_params(global_model)
                dictpfl_mgr.initialize_decomposition(init_params)
            else:
                dictpfl_mgr = None

            # Training loop
            for round_num in range(1, n_rounds + 1):
                tracker.start_round(round_num)

                global_params = get_model_params(global_model)
                for client_model in client_models:
                    set_model_params(client_model, global_params)

                client_gradients = []
                total_loss = 0.0

                for X_client, y_client in client_datasets:
                    idx = len(client_gradients)
                    loss = train_one_epoch(client_models[idx], X_client, y_client, lr=lr, batch_size=32)
                    total_loss += loss

                    updated_params = get_model_params(client_models[idx])
                    gradients = {name: updated_params[name] - global_params[name]
                               for name in global_params.keys()}
                    client_gradients.append(gradients)

                avg_loss = total_loss / n_clients

                # Aggregate
                if method_name == 'FedHE-Full':
                    aggregated_grads, method_metrics = fedhe_full_round(client_gradients, fhe)
                elif method_name == 'FedML-HE':
                    aggregated_grads, method_metrics = fedml_he_round(client_gradients, fhe, k_percent=0.1)
                elif method_name == 'DictPFL':
                    aggregated_grads, method_metrics = dictpfl_round(
                        client_gradients, fhe, dictpfl_mgr, prune_ratio=0.5, beta=0.2
                    )

                tracker.add_encryption_time(method_metrics['encryption_time'])
                tracker.add_decryption_time(method_metrics['decryption_time'])
                tracker.add_communication_bytes(method_metrics['communication_bytes'])

                # Update model
                for param_name in global_params.keys():
                    global_params[param_name] = global_params[param_name] + aggregated_grads[param_name]
                set_model_params(global_model, global_params)

                accuracy = evaluate_model(global_model, X_test, y_test)
                tracker.set_accuracy(accuracy)
                tracker.set_loss(avg_loss)
                tracker.end_round()

            results[method_name] = tracker
            status += f"{method_name}: Final accuracy = {tracker.get_summary()['final_accuracy']:.4f}\n"

        # Generate comparison plots
        methods_data = {name: tracker.get_round_data() for name, tracker in results.items()}
        summary_data = {name: tracker.get_summary() for name, tracker in results.items()}

        fig1 = plot_training_metrics(methods_data)
        fig2 = plot_efficiency_metrics(methods_data)
        fig3 = plot_comparison_bars(summary_data)
        fig4 = plot_encryption_breakdown(methods_data)

        # Add comparison summary to status
        status += f"\n{'='*50}\n"
        status += "COMPARISON SUMMARY\n"
        status += f"{'='*50}\n"

        comparison = ComparisonMetrics()
        for name, tracker in results.items():
            comparison.add_method(name, tracker)

        for name, summary in summary_data.items():
            status += f"\n{name}:\n"
            status += f"  Time: {summary['total_time']:.2f}s\n"
            status += f"  Communication: {summary['total_communication_mb']:.2f} MB\n"
            status += f"  Final Accuracy: {summary['final_accuracy']:.4f}\n"

        return status, fig1, fig2, fig3, fig4

    except Exception as e:
        return f"Error during comparison: {str(e)}", None, None, None, None


# Create Gradio interface
with gr.Blocks(title="DictPFL Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DictPFL: Efficient and Private Federated Learning Demo")
    gr.Markdown("""
    This demo compares three federated learning methods with homomorphic encryption:
    - **FedHE-Full**: Encrypt all gradients (full privacy, slowest)
    - **FedML-HE**: Encrypt top-10% gradients (partial privacy, faster)
    - **DictPFL**: Dictionary decomposition + pruning (full privacy, most efficient)
    """)

    with gr.Tab("System Setup"):
        gr.Markdown("## Initialize the Federated Learning System")

        with gr.Row():
            n_clients_input = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of Clients")
            n_samples_input = gr.Slider(minimum=500, maximum=5000, value=2000, step=500, label="Total Samples")
            seed_input = gr.Number(value=42, label="Random Seed")

        init_button = gr.Button("Initialize System", variant="primary")
        init_status = gr.Textbox(label="Initialization Status", lines=10)

        init_button.click(
            fn=initialize_system,
            inputs=[n_clients_input, n_samples_input, seed_input],
            outputs=init_status
        )

    with gr.Tab("Single Method Training"):
        gr.Markdown("## Train with a Single Method")

        with gr.Row():
            method_select = gr.Radio(
                choices=['FedHE-Full', 'FedML-HE', 'DictPFL'],
                value='DictPFL',
                label="Select Method"
            )
            n_rounds_single = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Rounds")
            lr_single = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="Learning Rate")

        with gr.Row():
            k_percent_input = gr.Slider(minimum=0.05, maximum=0.5, value=0.1, step=0.05,
                                       label="FedML-HE: Top-k% to Encrypt")
            prune_ratio_input = gr.Slider(minimum=0.3, maximum=0.7, value=0.5, step=0.1,
                                         label="DictPFL: Pruning Ratio")
            beta_input = gr.Slider(minimum=0.0, maximum=0.5, value=0.2, step=0.1,
                                  label="DictPFL: Reactivation Probability")

        run_single_button = gr.Button("Run Training", variant="primary")

        single_status = gr.Textbox(label="Training Status", lines=15)

        with gr.Row():
            single_plot1 = gr.Plot(label="Training Metrics")
            single_plot2 = gr.Plot(label="Efficiency Metrics")

        run_single_button.click(
            fn=run_single_method,
            inputs=[method_select, n_rounds_single, lr_single, k_percent_input, prune_ratio_input, beta_input],
            outputs=[single_status, single_plot1, single_plot2]
        )

    with gr.Tab("Method Comparison"):
        gr.Markdown("## Compare All Three Methods")

        with gr.Row():
            n_rounds_comp = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Rounds")
            lr_comp = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="Learning Rate")

        run_comp_button = gr.Button("Run Comparison", variant="primary")

        comp_status = gr.Textbox(label="Comparison Status", lines=20)

        with gr.Row():
            comp_plot1 = gr.Plot(label="Training Metrics Comparison")
            comp_plot2 = gr.Plot(label="Efficiency Metrics Comparison")

        with gr.Row():
            comp_plot3 = gr.Plot(label="Summary Comparison")
            comp_plot4 = gr.Plot(label="Encryption Breakdown")

        run_comp_button.click(
            fn=run_comparison,
            inputs=[n_rounds_comp, lr_comp],
            outputs=[comp_status, comp_plot1, comp_plot2, comp_plot3, comp_plot4]
        )

    gr.Markdown("""
    ---
    **Note**: FHE operations are computationally intensive. Training may take several minutes depending on your hardware.

    **Paper**: DictPFL: Efficient and Private Federated Learning on Encrypted Gradients
    """)


if __name__ == "__main__":
    demo.launch(share=False)
