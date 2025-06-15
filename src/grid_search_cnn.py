import itertools
import subprocess
import os
import yaml
from tqdm import tqdm

os.environ["WANDB_SILENT"] = "true"

# Define grid search options
models = ["CNN", "ImprovedCNN", "DeepCNN"]
batch_sizes = [32, 64, 128]
learning_rates = [1e-3]
cnn_channels_list = [64, 128, 256]
kernel_sizes = [3, 5, 7]

all_combinations = list(itertools.product(models, batch_sizes, learning_rates, cnn_channels_list, kernel_sizes))

for model_type, bs, lr, ch, ks in tqdm(all_combinations, desc="Grid Search Progress"):
    run_name = f"{model_type.lower()}-bs-{bs}-cnn_c-{ch}-k-{ks}"
    config = {
        "run_name": run_name,
        "log_dir": "./logs",
        "batch_size": bs,
        "epochs": 100,  # Increased max epochs since we have early stopping
        "lr": lr,
        "cnn_channels": ch,
        "kernel_size": ks,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "model_type": model_type
    }

    # Check if result file exists
    result_file = os.path.join("logs", f"{run_name}.json")  # Changed to .json since we save history as json
    if os.path.exists(result_file):
        tqdm.write(f"Skipping {run_name} (result exists: {result_file})")
        continue

    # Write config to YAML
    with open("src/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # tqdm.write(f"Running: {run_name}")
    subprocess.run(["python", "src/train_cnn.py"])