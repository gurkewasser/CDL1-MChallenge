import itertools
import subprocess
import os

# Define grid search options
batch_sizes = [8, 16, 32, 64]
learning_rates = [1e-3, 1e-4, 5e-4, 1e-5]
cnn_channels_list = [16, 32, 64, 128]
kernel_sizes = [3, 5, 7]

for bs, lr, ch, ks in itertools.product(batch_sizes, learning_rates, cnn_channels_list, kernel_sizes):
    run_name = f"cnn-bs-{bs}-{lr}-cnn_c-{ch}-k-{ks}"
    config = {
        "project": "cdl1",
        "run_name": run_name,
        "use_wandb": True,
        "log_dir": "./logs",
        "batch_size": bs,
        "epochs": 10,
        "lr": lr,
        "cnn_channels": ch,
        "kernel_size": ks,
    }

    # Check if result file exists
    result_file = os.path.join("logs", f"{run_name}.pth")
    if os.path.exists(result_file):
        print(f"Skipping {run_name} (result exists: {result_file})")
        continue

    # Write config to YAML
    with open("src/config.yaml", "w") as f:
        import yaml
        yaml.safe_dump(config, f)

    print(f"Running: {run_name}")
    subprocess.run(["python", "src/train_cnn.py"])