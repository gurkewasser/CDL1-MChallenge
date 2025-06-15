import itertools
import subprocess
import os

# Add tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback: no progress bar

# Define grid search options
batch_sizes = [16, 32, 64]
learning_rates = [1e-3, 5e-4, 1e-4]
hidden_sizes_combinations = [
    [256, 128, 64],  # Shallow network
    [512, 256, 128],  # Medium network
    [1024, 512, 256],  # Deep network
]
dropouts = [0.2, 0.3, 0.4]
model_types = ["basic", "advanced"]

# Prepare all combinations in advance for progress bar
all_combinations = list(itertools.product(
    batch_sizes, learning_rates, hidden_sizes_combinations, dropouts, model_types
))

for bs, lr, hs, do, mt in tqdm(all_combinations, desc="Grid Search MLP", unit="run"):
    run_name = f"mlp-bs{bs}-lr{lr}-hs{hs[0]}-do{do}-{mt}"
    config = {
        # Model Architecture
        "model_type": mt,
        "mlp_hidden_sizes": hs,
        "dropout": do,

        # Training Parameters
        "batch_size": bs,
        "epochs": 100,
        "lr": lr,
        "weight_decay": 1e-5,

        # Data Parameters
        "train_val_split": 0.8,
        "val_test_split": 0.1,

        # Logging and Monitoring
        "use_wandb": False,  # Disable wandb logging in terminal
        "project": "CDL1-MLP",
        "run_name": run_name,
        "log_dir": "logs/mlp",

        # Early Stopping
        "early_stopping": True,
        "patience": 15,
        "min_delta": 0.0005,
        "restore_best_weights": True,

        # Learning Rate Scheduling
        "use_lr_scheduler": True,
        "lr_scheduler": "reduce_on_plateau",
        "lr_patience": 7,
        "lr_factor": 0.2,
        "min_lr": 1e-7,
        "warmup_epochs": 5,

        # Model Checkpointing
        "save_best_only": True,
        "checkpoint_dir": "checkpoints/mlp",
        "save_frequency": 5
    }

    # Check if result file exists
    result_file = os.path.join("logs/mlp", f"{run_name}.json")
    if os.path.exists(result_file):
        print(f"Skipping {run_name} (result exists: {result_file})")
        continue

    # Create necessary directories
    os.makedirs("logs/mlp", exist_ok=True)
    os.makedirs("checkpoints/mlp", exist_ok=True)

    # Write config to YAML
    with open("src/config_mlp.yaml", "w") as f:
        import yaml
        yaml.safe_dump(config, f)

    print(f"Running: {run_name}")
    subprocess.run(["python", "src/train_mlp.py"]) 