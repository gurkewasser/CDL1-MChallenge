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
import pandas as pd
import sys

# Load configuration
with open("src/config_lstm.yaml", "r") as f:
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

class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of the fully connected layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class AdvancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of the fully connected layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final classification
        out = self.fc(context)
        return out

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, epoch, val_loss, model):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.get("patience", 10),
        min_delta=config.get("min_delta", 0.001),
        mode='min'
    )
    
    # Initialize learning rate scheduler if enabled
    if config.get("use_lr_scheduler", False):
        if config["lr_scheduler"] == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.get("lr_factor", 0.5),
                patience=config.get("lr_patience", 5),
                min_lr=config.get("min_lr", 1e-6)
            )
        elif config["lr_scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["epochs"],
                eta_min=config.get("min_lr", 1e-6)
            )
        else:  # step scheduler
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("lr_step_size", 10),
                gamma=config.get("lr_factor", 0.5)
            )

    # Learning rate warmup
    if config.get("warmup_epochs", 0) > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config["warmup_epochs"]
        )

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

        # Update learning rate scheduler
        if config.get("use_lr_scheduler", False):
            if config["lr_scheduler"] == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Update warmup scheduler if in warmup phase
        if config.get("warmup_epochs", 0) > 0 and epoch < config["warmup_epochs"]:
            warmup_scheduler.step()

        #print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
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
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Early stopping check
        if early_stopping(epoch, val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best model was at epoch {early_stopping.best_epoch+1} with validation loss: {-early_stopping.best_score:.4f}")
            break

        # Save periodic checkpoint
        if config.get("save_frequency", 0) > 0 and (epoch + 1) % config["save_frequency"] == 0:
            checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/lstm"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{config['run_name']}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

    # Restore best model
    model.load_state_dict(early_stopping.best_model_state)
    
    # Save best model if checkpointing is enabled
    if config.get("save_best_only", True):
        checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/lstm"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{config['run_name']}_best.pth"
        torch.save({
            'epoch': early_stopping.best_epoch,
            'model_state_dict': early_stopping.best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': -early_stopping.best_score,
        }, checkpoint_path)
        print(f"Best model saved to {checkpoint_path}")

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

# Choose model architecture
if config.get("model_type", "basic") == "advanced":
    model = AdvancedLSTM(
        input_size=X.shape[2],
        hidden_size=config["lstm_hidden_size"],
        num_layers=config["lstm_num_layers"],
        num_classes=num_classes,
        dropout=config.get("dropout", 0.2),
        bidirectional=config.get("bidirectional", False)
    ).to(device)
else:
    model = BasicLSTM(
        input_size=X.shape[2],
        hidden_size=config["lstm_hidden_size"],
        num_layers=config["lstm_num_layers"],
        num_classes=num_classes,
        dropout=config.get("dropout", 0.2),
        bidirectional=config.get("bidirectional", False)
    ).to(device)

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