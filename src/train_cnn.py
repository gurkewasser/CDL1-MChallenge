import os, json, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import sys
from tqdm import trange
#from preprocess_functions import compute_normalization_params, apply_normalization

# --- wandb logging ---
import wandb

# Load configuration
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set up paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root.parent))

from methods import compute_normalization_params, apply_normalization

train_dir = Path("data/processed/DL")
train_files = sorted(train_dir.glob("dl_train*.npz"))
train_arrays = [np.load(f, allow_pickle=True) for f in train_files]
X = np.concatenate([arr["X"] for arr in train_arrays], axis=0)
y = np.concatenate([arr["y"] for arr in train_arrays], axis=0)

# Load validation set from dl_val.npz
data_val = np.load("data/processed/DL/dl_val.npz")
X_val = data_val["X"]
y_val = data_val["y"]

data_test = np.load("data/processed/DL/dl_test.npz")

# Encode labels if necessary
if y.dtype.kind in {'U', 'S', 'O'} or not np.issubdtype(y.dtype, np.integer):
    le = LabelEncoder()
    y = le.fit_transform(y)
    y_val = le.transform(y_val)
else:
    le = None
    y_val = y_val.astype(np.int64)

# Ensure shape (N, C, T)
if X.shape[1] < X.shape[2]:
    X = np.transpose(X, (0, 2, 1))
if X_val.shape[1] < X_val.shape[2]:
    X_val = np.transpose(X_val, (0, 2, 1))

X = X.astype(np.float32)
y = y.astype(np.int64)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.int64)

# Apply normalization based on training data
stats = compute_normalization_params(pd.DataFrame(X.reshape(X.shape[0], -1)))
X_flat = X.reshape(X.shape[0], -1)
X_norm = apply_normalization(pd.DataFrame(X_flat), stats).values.reshape(X.shape)
X = X_norm

X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_val_norm = apply_normalization(pd.DataFrame(X_val_flat), stats).values.reshape(X_val.shape)
X_val = X_val_norm

num_classes = len(np.unique(y))

# Dataset setup
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
train_set = TensorDataset(X_tensor, y_tensor)

X_val_tensor = torch.from_numpy(X_val)
y_val_tensor = torch.from_numpy(y_val)
val_set = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

# Prepare test_loader
X_test = data_test["X"]
y_test = data_test["y"]

if X_test.shape[1] < X_test.shape[2]:
    X_test = np.transpose(X_test, (0, 2, 1))

X_test = X_test.astype(np.float32)
if le is not None:
    y_test = le.transform(y_test)
else:
    y_test = y_test.astype(np.int64)

X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_test_norm = apply_normalization(pd.DataFrame(X_test_flat), stats).values.reshape(X_test.shape)
X_test = X_test_norm

X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)
test_set = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

# CNN Modell
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class ImprovedCNN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, kernel_size=5):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class DeepCNN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, kernel_size=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels * 4, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_channels * 4, num_classes)

    def forward(self, x):
        residual = self.shortcut(x).squeeze(-1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x).squeeze(-1)
        x = x + residual  # Residual connection
        x = self.dropout(x)
        x = self.fc(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, value, model, optimizer):
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_epoch = epoch
                return False
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_epoch = epoch
                return False

        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
        mode='min'
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        val_loss = None
        val_acc = None
        if val_loader is not None:
            # Validation
            model.eval()
            val_loss = 0
            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    preds.append(out.argmax(dim=1).cpu().numpy())
                    labels.append(yb.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))

            # Update learning rate
            scheduler.step(val_loss)
        else:
            # No validation loader, step scheduler with train loss
            scheduler.step(train_loss)

        # wandb logging
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
        if val_loss is not None:
            log_dict["val_loss"] = val_loss
        if val_acc is not None:
            log_dict["val_acc"] = val_acc

        wandb.log(log_dict)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        if val_loss is not None:
            history["val_loss"].append(val_loss)
        if val_acc is not None:
            history["val_acc"].append(val_acc)

        # Early stopping check
        if val_loss is not None and early_stopping(epoch, val_loss, model, optimizer):
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best epoch was {early_stopping.best_epoch+1} with validation loss {early_stopping.best_value:.4f}")
            break

    return model, history

# Evaluation
def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    #print(f"\nTest Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1, y_true, y_pred

# Save history
def save_history(history, config, test_metrics):
    os.makedirs(config["log_dir"], exist_ok=True)
    history_path = Path(config["log_dir"]) / f"{config['run_name']}.json"
    history["test_metrics"] = {
        "test_accuracy": test_metrics[0],
        "test_precision": test_metrics[1],
        "test_recall": test_metrics[2],
        "test_f1": test_metrics[3],
        "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": history["val_acc"][-1] if history["val_acc"] else None
    }
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # wandb: log test metrics as summary
    wandb.summary["test_accuracy"] = test_metrics[0]
    wandb.summary["test_precision"] = test_metrics[1]
    wandb.summary["test_recall"] = test_metrics[2]
    wandb.summary["test_f1"] = test_metrics[3]

# Main Run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb: initialize run
wandb.init(
    project=config.get("wandb_project", "cnn_gridsearch_cdl1"),
    name=config.get("run_name", None),
    config=config,
    dir=config.get("log_dir", "./logs"),
    reinit=True,
)

# Select model based on config
if config["model_type"] == "CNN":
    model = CNN(in_channels=X.shape[1], out_channels=config["cnn_channels"], 
                kernel_size=config["kernel_size"], num_classes=num_classes).to(device)
elif config["model_type"] == "ImprovedCNN":
    model = ImprovedCNN(in_channels=X.shape[1], num_classes=num_classes, 
                       hidden_channels=config["cnn_channels"], 
                       kernel_size=config["kernel_size"]).to(device)
else:  # DeepCNN
    model = DeepCNN(in_channels=X.shape[1], num_classes=num_classes, 
                   hidden_channels=config["cnn_channels"], 
                   kernel_size=config["kernel_size"]).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss()

model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device)

# Evaluate only if test_loader has data
if len(test_loader.dataset) > 0:
    acc, prec, rec, f1, y_true, y_pred = evaluate(model, test_loader, device)
    wandb.log({
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "epoch": history["epoch"][-1] if history["epoch"] else config["epochs"],
        "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "device": str(device),
        "model_type": config["model_type"],
    })
    save_history(history, config, test_metrics=(acc, prec, rec, f1))
else:
    print("⚠️ Kein Test-Set vorhanden – Evaluation wird übersprungen.")
    save_history(history, config, test_metrics=(None, None, None, None))

wandb.finish()