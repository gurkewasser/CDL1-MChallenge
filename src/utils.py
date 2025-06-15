from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred, average='macro'):
    """
    Computes evaluation metrics for the model predictions.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        average (str): Type of averaging to be performed on the data.
                       Can be 'micro', 'macro', 'samples', 'weighted'.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def print_compute_metrics(metrics, average='macro'):
    print('---Model Evaluation---')
    print(f'accuracy = {metrics["accuracy"]}\n')
    print(f'precision = {metrics["precision"]}\n')
    print(f'recall = {metrics["recall"]}\n')
    print(f'f1 = {metrics["f1"]}\n')
    print('-----------------------')
    


def plot_metrics(metrics_dict, title='Model Evaluation Metrics'):
    """
    Plots the evaluation metrics as a bar chart.
    Args:
        metrics_dict (dict): Dictionary containing the metrics to plot.
        title (str): Title of the plot.
    """
    names = list(metrics_dict.keys())
    values = [metrics_dict[name] for name in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title(title)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.5f}", ha='center', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy',
                        title='Learning Curve', train_sizes=np.linspace(0.1, 1.0, 5),
                        n_jobs=-1, figsize=(8, 5)):
    """
    Plottet eine Lernkurve für das gegebene Modell (Estimator).

    Parameter:
    - estimator: trainiertes Modell oder Pipeline (z. B. grid.best_estimator_)
    - X: Feature-Matrix
    - y: Zielvariable
    - cv: Anzahl der Cross-Validation-Splits
    - scoring: Bewertungsmetrik ('accuracy', 'f1', etc.)
    - title: Titel des Plots
    - train_sizes: Größenanteile der Trainingsdaten
    - n_jobs: parallele Prozesse (Standard: alle)
    - figsize: Größe des Plots
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=n_jobs
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, 's-', label='Cross-Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title(title)
    plt.xlabel('Anzahl Trainingsbeispiele')
    plt.ylabel(scoring.capitalize())
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_no_data_leakage(df, X_train, X_test, y_train, y_test):
    """
    Überprüft, ob nach train_test_split kein Datenleck vorliegt:
    1. Die Indizes von X_train und X_test dürfen sich nicht überlappen.
    2. Die Indizes von y_train und y_test dürfen sich nicht überlappen.
    3. Die vereinigten Indizes (Train ∪ Test) müssen genau den Indizes des Originals entsprechen.
    4. Die Spalten von X_train und X_test sollten identisch sein.
    
    Wir gehen davon aus, dass df der ursprüngliche DataFrame ist, aus dem X und y abgeleitet wurden.
    """
    # 1. Überlappung der Indizes prüfen
    overlap_X = X_train.index.intersection(X_test.index)
    overlap_y = y_train.index.intersection(y_test.index)
    assert overlap_X.empty, f"Datenleck in Features: gemeinsame Indizes {overlap_X.tolist()}"
    assert overlap_y.empty, f"Datenleck in Zielvariablen: gemeinsame Indizes {overlap_y.tolist()}"
    
    # 2. Vereinigung der Indizes muss die Original-Indizes ergeben
    train_test_indices = X_train.index.union(X_test.index)
    original_indices = df.index
    assert set(train_test_indices) == set(original_indices), (
        "Die vereinigten Indizes von Train und Test stimmen nicht mit den Original-Indizes überein."
    )
    
    # 3. Spalten von X_train und X_test müssen identisch sein
    assert list(X_train.columns) == list(X_test.columns), (
        "Die Spalten von X_train und X_test weichen voneinander ab."
    )
    
    print("✅ Kein Data Leakage entdeckt")



def plot_performance_metrics(accuracy, precision, recall, f1, dataset_name: str):
    """
    Zeichnet ein modernes Balkendiagramm für vier Performance-Metriken.
    
    Parameter:
    -----------
    accuracy : float
        Accuracy-Wert (zwischen 0 und 1).
    precision : float
        Precision-Wert (zwischen 0 und 1).
    recall : float
        Recall-Wert (zwischen 0 und 1).
    f1 : float
        F1-Score-Wert (zwischen 0 und 1).
    dataset_name : str, optional
        Bezeichnung der Daten (z.B. "Trainingsdaten" oder "Testdaten"), 
        die im Titel angezeigt wird.
    """
    metriken_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metriken_werte = [accuracy, precision, recall, f1]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Balken zeichnen
    bars = ax.bar(metriken_names, metriken_werte, edgecolor='black', linewidth=1)

    # 1) Spines minimieren: oben, rechts, links ausblenden
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    # Untere Achse bleibt erhalten
    ax.spines['bottom'].set_color('#333333')

    # 2) Dezentes Gitternetz auf der y-Achse
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)  # Gitternetz hinter die Balken legen

    # 3) y-Achsen-Ticks verschlanken (kein Strich, nur Zahlen)
    ax.tick_params(axis='y', length=0)

    # 4) Werte (mit zwei Nachkommastellen) über den Balken anzeigen
    for bar, wert in zip(bars, metriken_werte):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{wert:.4f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    # 5) Achsentitel, Begrenzungen und Schriftgrößen
    ax.set_ylabel("Wert", fontsize=12)
    ax.set_ylim(0, 1.05)  # etwas Abstand über 1.0 lassen
    ax.set_title(f"Performance-Metriken – Random Forest ({dataset_name})", fontsize=14, pad=12)

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.show()





def plot_model_metrics(models_to_plot, all_data, max_models=None, figsize_per_model=(14, 4), show_grid=True):
    """
    Plottet Training/Validation Loss und Validation Accuracy für eine Liste von Modell-Runs.
    
    Args:
        models_to_plot (pd.DataFrame): DataFrame mit mindestens der Spalte 'run_name'.
        all_data (dict): Dict mit run_name als Key und Werten als Dicts mit 'epoch', 'train_loss', 'val_loss', 'val_acc'.
        max_models (int, optional): Max. Anzahl an Modellen, die geplottet werden sollen. None = alle.
        figsize_per_model (tuple, optional): Breite und Höhe pro Modell-Zeile (default: (14, 4)).
        show_grid (bool): Wenn True, wird Grid auf Plots angezeigt.
    """
    top_n = len(models_to_plot) if max_models is None else min(max_models, len(models_to_plot))
    
    fig, axes = plt.subplots(top_n, 2, figsize=(figsize_per_model[0], figsize_per_model[1] * top_n))
    
    # Sicherstellen, dass axes 2D ist
    if top_n == 1:
        axes = axes.reshape(1, 2)
    
    for i, (_, row) in enumerate(models_to_plot.head(top_n).iterrows()):
        run_name = row["run_name"]
        data = all_data[run_name]
        epochs = data["epoch"]
        
        # Plot Loss
        ax_loss = axes[i, 0]
        ax_loss.plot(epochs, data["train_loss"], label="Train Loss")
        ax_loss.plot(epochs, data["val_loss"], label="Val Loss")
        ax_loss.set_title(f"{run_name} - Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        if show_grid:
            ax_loss.grid(True)
        
        # Plot Accuracy
        ax_acc = axes[i, 1]
        ax_acc.plot(epochs, data["val_acc"], label="Val Accuracy")
        ax_acc.set_title(f"{run_name} - Validation Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        if show_grid:
            ax_acc.grid(True)
    
    plt.tight_layout()
    plt.show()

