import itertools
import subprocess
import os
from tqdm import tqdm

# Define grid search options
batch_sizes = [16, 32]
learning_rates = [1e-3, 1e-4]
hidden_sizes = [64, 128]
num_layers = [1, 2]
dropouts = [0.2, 0.5]
bidirectional_options = [True, False]
model_types = ["basic", "advanced"]

# Prepare all combinations in advance for progress bar
all_combinations = list(itertools.product(
    model_types, batch_sizes, learning_rates, hidden_sizes, num_layers, dropouts, bidirectional_options
))

for mt, bs, lr, hs, nl, do, bi in tqdm(all_combinations, desc="Grid Search LSTM", unit="run"):
    run_name = f"{mt}-bs{bs}-lr{lr}-hs{hs}-nl{nl}-do{do}-bi{bi}"
    config = {
        # Model Architecture
        "model_type": mt,
        "lstm_hidden_size": hs,
        "lstm_num_layers": nl,
        "bidirectional": bi,
        "dropout": do,

        # Training Parameters
        "batch_size": bs,
        "epochs": 100,  # Increased epochs
        "lr": lr,
        "weight_decay": 1e-5,

        # Data Parameters
        "train_val_split": 0.8,
        "val_test_split": 0.1,

        # Logging and Monitoring
        "use_wandb": False,  # Disable wandb logging in terminal
        "project": "CDL1-LSTM",
        "run_name": run_name,
        "log_dir": "logs/lstm",

        # Early Stopping
        "early_stopping": True,
        "patience": 15,  # Increased patience
        "min_delta": 0.0005,  # More sensitive to improvements
        "restore_best_weights": True,  # Ensure we keep the best model

        # Learning Rate Scheduling
        "use_lr_scheduler": True,
        "lr_scheduler": "reduce_on_plateau",
        "lr_patience": 7,  # Increased patience for LR reduction
        "lr_factor": 0.2,  # More aggressive LR reduction
        "min_lr": 1e-7,  # Lower minimum learning rate
        "warmup_epochs": 5,  # Learning rate warmup

        # Model Checkpointing
        "save_best_only": True,
        "checkpoint_dir": "checkpoints/lstm",
        "save_frequency": 5  # Save checkpoints every 5 epochs
    }

    # Check if result file exists
    result_file = os.path.join("logs/lstm", f"{run_name}.json")
    if os.path.exists(result_file):
        print(f"Skipping {run_name} (result exists: {result_file})")
        continue

    # Create necessary directories
    os.makedirs("logs/lstm", exist_ok=True)
    os.makedirs("checkpoints/lstm", exist_ok=True)

    # Write config to YAML
    with open("src/config_lstm.yaml", "w") as f:
        import yaml
        yaml.safe_dump(config, f)

    print(f"Running: {run_name}")
    subprocess.run(["python", "src/train_lstm.py"]) 