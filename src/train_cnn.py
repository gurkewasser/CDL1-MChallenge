import os, json, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import wandb
from pathlib import Path

# Load configuration
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set up paths
import rootutils
root = rootutils.setup_root(search_from=".", indicator=".git")
data_train = np.load(root / "data/DL/TRAIN/dl_data_train.npz", allow_pickle=True)
data_test = np.load(root / "data/DL/TEST/dl_data_test.npz", allow_pickle=True)

X = data_train["X"]
y = data_train["y"]

# Encode labels if necessary
if y.dtype.kind in {'U', 'S', 'O'} or not np.issubdtype(y.dtype, np.integer):
    le = LabelEncoder()
    y = le.fit_transform(y)
else:
    le = None

# Ensure shape (N, C, T)
if X.shape[1] < X.shape[2]:
    X = np.transpose(X, (0, 2, 1))

X = X.astype(np.float32)
y = y.astype(np.int64)
num_classes = len(np.unique(y))

# Dataset split
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
dataset = TensorDataset(X_tensor, y_tensor)
n = len(dataset)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
n_test = n - n_train - n_val
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

# Dataloaders
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"])
test_loader = DataLoader(test_set, batch_size=config["batch_size"])

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

# Training
def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(config["epochs"]):
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

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if config["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
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
    print(f"\nTest Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
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
    }
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

# Main Run
if config["use_wandb"]:
    wandb.init(project=config["project"], name=config["run_name"], config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=X.shape[1], out_channels=config["cnn_channels"], kernel_size=config["kernel_size"], num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss()

model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device)
acc, prec, rec, f1, y_true, y_pred = evaluate(model, test_loader, device)

if config["use_wandb"]:
    wandb.log({
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
    })
    wandb.finish()

save_history(history, config, test_metrics=(acc, prec, rec, f1))