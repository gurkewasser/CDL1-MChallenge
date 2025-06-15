from sklearn.model_selection import train_test_split
import numpy as np
import zipfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import pandas as pd
from zipfile import ZipFile
import tempfile

SENSOR_FILES = {
    "Accelerometer.csv": ["x", "y", "z"],
    "Magnetometer.csv": ["x", "y", "z"],
    "Gravity.csv": ["x", "y", "z"],
    "Gyroscope.csv": ["x", "y", "z"],
    "Orientation.csv": ["pitch", "roll", "yaw", "qw", "qx", "qy", "qz"],
}

def load_and_merge_sensor_data(zip_path: str) -> pd.DataFrame:
    zip_path = Path(zip_path)
    assert zip_path.exists(), f"ZIP not found: {zip_path}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        dfs = []
        for filename, columns in SENSOR_FILES.items():
            file_path = Path(tmpdir) / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                df = df[["seconds_elapsed"] + columns]
                df.columns = ["time"] + [f"{filename[:-4].lower()}_{col}" for col in columns]
                # Convert "time" column to datetime using nanoseconds
                dfs.append(df)
        
        # Merge über Zeitstempel
        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = pd.merge_asof(df_merged.sort_values("time"), df.sort_values("time"), on="time", direction="nearest")
    
    df_merged["timestamp"] = pd.to_datetime(df_merged["time"], unit='s')

    return df_merged

def compute_normalization_params(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns
    params = {
        col: {
            "mean": df[col].mean(),
            "std": df[col].std(ddof=0) + 1e-8  # kleine Konstante zur Vermeidung von Division durch 0
        }
        for col in numeric_cols
    }
    return params

def apply_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    for col, stats in params.items():
        if col in df.columns:
            df[col] = (df[col] - stats["mean"]) / stats["std"]
    return df

def combine_sessions(zip_list, augment=False, augment_noise_level=0):
    all_dfs = []

    for zip_file in zip_list:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            df = load_and_merge_sensor_data(zip_file)

            metadata_path = Path(tmpdir) / "Metadata.csv"
            df = standardize_coordinates(df, metadata_path)

            if augment:
                df = apply_augmentation(df, noise_level=augment_noise_level)

            df = preprocess_sampling(df, target_hz=50)
            df["label"] = get_label_from_filename(zip_file.name)
            print(f"zip_file: {zip_file.name}")
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

def apply_augmentation(df: pd.DataFrame, noise_level: float = 0.2) -> pd.DataFrame:
    """
    Fügt Rauschen zu numerischen Sensordaten hinzu.
    noise_level = Standardabweichung des Rauschens.
    """
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if col != "time":  # Zeitspalte nicht augmentieren
            df[col] += np.random.normal(0, noise_level, size=len(df))
    return df

def get_label_from_filename(filename: str) -> str:
    return Path(filename).stem.split("_")[0]

def create_stratified_split(zip_files: list, test_size: float = 0.2, seed: int = 42):
    """
    Erstellt einen stratified Split für eine Liste von ZIP-Dateipfaden.
    zip_files: Liste von Pfaden (Path-Objekte oder Strings)
    """
    if not zip_files:
        print("⚠️ Keine ZIP-Dateien gefunden. Rückgabe: leere Listen.")
        return [], []

    # Sicherstellen, dass wir mit Path-Objekten arbeiten
    zip_files = [Path(f) for f in zip_files]

    data = pd.DataFrame({
        "path": zip_files,
        "label": [get_label_from_filename(f.name) for f in zip_files]
    })

    label_counts = data["label"].value_counts()
    if len(label_counts) < 2:
        print(f"⚠️ Zu wenige verschiedene Labels ({label_counts.index.tolist()}) für Stratified Split. Rückgabe: leere Listen.")
        return [], []
    if (label_counts < 2).any():
        print(f"⚠️ Mindestens ein Label kommt nur einmal vor: {label_counts[label_counts < 2].index.tolist()}. Stratified Split nicht möglich. Rückgabe: leere Listen.")
        return [], []

    try:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            data, test_size=test_size, stratify=data["label"], random_state=seed
        )
    except Exception as e:
        print(f"⚠️ Fehler beim Stratified Split: {e}. Rückgabe: leere Listen.")
        return [], []

    return train_df["path"].tolist(), test_df["path"].tolist()

def standardize_coordinates(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    """
    Erkennt automatisch iOS vs. Android aus Metadata.csv und bringt Daten in den Android-Rahmen.
    Nur nötig, wenn Standardisierung in der App deaktiviert war.
    """

    platform = "unknown"
    if metadata_path and metadata_path.exists():
        try:
            platform = pd.read_csv(metadata_path)["platform"].iloc[0].lower()
        except Exception as e:
            print(f"⚠️ Fehler beim Lesen von {metadata_path}: {e}")
    else:
        print(f"⚠️ Konnte Plattform nicht auslesen: {metadata_path}")

    df = df.copy()

    try:
        metadata = pd.read_csv(metadata_path)
        platform = metadata["platform"].iloc[0].strip().lower()  # z. B. "ios" oder "android"
    except Exception as e:
        print(f"⚠️ Konnte Plattform nicht auslesen: {e}")
        platform = "unknown"

    if platform == "ios":
        # iOS → Android-Konvertierung (siehe COORDINATES.md)
        for sensor in ["accelerometer", "gyroscope", "gravity", "magnetometer"]:
            x_col = f"{sensor}_x"
            if x_col in df.columns:
                df[x_col] *= -1
        if "orientation_roll" in df.columns:
            df["orientation_roll"] *= -1

    return df

def segment_dataframe(df: pd.DataFrame, window_size: int = 250, stride: int = 125) -> list[dict]:
    """
    Schneidet die Zeitreihe in überlappende Segmente (z. B. 5s bei 50Hz = 250 Samples).
    Gibt Liste von Dicts mit {'data': ndarray, 'label': str, 'feature_names': list}.
    """
    segments = []

    # Wähle nur numerische Spalten außer 'time' oder 'timestamp'
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [col for col in numeric_cols if col not in ["time", "timestamp"]]

    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window = df.iloc[start:end]

        # Prüfen auf Mindestlänge (zur Sicherheit)
        if len(window) < window_size:
            continue

        segment_data = window[numeric_cols].to_numpy()
        label = window["label"].mode()[0]  # häufigstes Label im Segment
        
        segments.append({
            "data": segment_data, 
            "label": label,
            "feature_names": list(numeric_cols) 
        })

    return segments

def trim_data(df: pd.DataFrame, trim_seconds: float = 8.0) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time")
    max_time = df["time"].max()
    df = df[(df["time"] >= trim_seconds) & (df["time"] <= (max_time - trim_seconds))]
    return df

def resample_data(df: pd.DataFrame, target_hz: int = 50) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time")

    new_times = np.arange(df["time"].min(), df["time"].max(), 1 / target_hz)

    df_interp = df.set_index("time")
    df_interp = df_interp.infer_objects(copy=False)
    df_interp = df_interp.interpolate(method="linear").reset_index()

    df_resampled = pd.DataFrame({"time": new_times})
    df_resampled = pd.merge_asof(df_resampled, df_interp, on="time", direction="nearest")

    df_resampled["timestamp"] = pd.to_datetime(df_resampled["time"], unit="s")
    return df_resampled

def preprocess_sampling(df: pd.DataFrame, target_hz: int = 50) -> pd.DataFrame:
    df_trimmed = trim_data(df)
    df_resampled = resample_data(df_trimmed, target_hz=target_hz)
    return df_resampled

def apply_moving_average(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if col not in ["time"]:  # 'timestamp' kannst du auch ausschließen
            df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
    return df

def plot_accelerometer(df, title="Accelerometer", seconds: float = None):
    df_plot = df

    if seconds is not None:
        if "time" in df.columns:
            df_plot = df[df["time"] <= seconds]
        elif "timestamp" in df.columns:
            min_time = df["timestamp"].min()
            df_plot = df[df["timestamp"] <= (min_time + pd.Timedelta(seconds=seconds))]
        # else: fallback to all data

    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["timestamp"], df_plot["accelerometer_x"], label="accelerometer_x", color="red")
    plt.plot(df_plot["timestamp"], df_plot["accelerometer_y"], label="accelerometer_y", color="green")
    plt.plot(df_plot["timestamp"], df_plot["accelerometer_z"], label="accelerometer_z", color="blue")
    plt.title(title)
    plt.xlabel("Zeit (s)")
    plt.ylabel("Beschleunigung (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_segment(segment, title="Segment – Accelerometer"):
    data = segment["data"]
    label = segment["label"]
    
    plt.figure(figsize=(12, 4))
    plt.plot(data[:, 0], label="acc_x", color="red")  # angenommen, Spalte 0 = accelerometer_x
    plt.plot(data[:, 1], label="acc_y", color="green")
    plt.plot(data[:, 2], label="acc_z", color="blue")
    plt.title(f"{title} | Label: {label}")
    plt.xlabel("Zeit (Samples)")
    plt.ylabel("Beschleunigung (normalisiert)")
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_features_from_segments(segments: list[dict], feature_names: list[str] = None) -> pd.DataFrame:
    """
    Extrahiert statistische Merkmale aus Segmenten für klassische ML-Modelle.
    Wenn feature_names angegeben wird, werden die Feature-Namen direkt verwendet (z. B. 'accelerometer_x').
    """
    feature_list = []

    for seg in segments:
        data = seg["data"]  # shape: (window_size, num_features)
        label = seg["label"]
        features = {}

        for i in range(data.shape[1]):
            col = data[:, i]

            # Feature-Basename: z. B. 'accelerometer_x' statt 'f0'
            base = feature_names[i] if feature_names else f"f{i}"

            features[f"{base}_mean"] = np.mean(col)
            features[f"{base}_std"] = np.std(col)
            features[f"{base}_min"] = np.min(col)
            features[f"{base}_max"] = np.max(col)

        features["label"] = label
        feature_list.append(features)

    return pd.DataFrame(feature_list)


def augment_orientation_pose(df: pd.DataFrame, method: str = "flip_roll") -> pd.DataFrame:
    """
    Simuliert alternative Handyhaltungen über die Orientation-Daten.
    Methoden:
    - 'flip_roll': Roll umkehren (z. B. Seitenwechsel)
    - 'invert_pitch': Pitch spiegeln (z. B. Display nach innen statt außen)
    - 'rotate_yaw_180': Yaw um 180° drehen (z. B. Handy auf dem Kopf)
    """
    df = df.copy()
    
    if method == "flip_roll" and "orientation_roll" in df.columns:
        df["orientation_roll"] *= -1

    elif method == "invert_pitch" and "orientation_pitch" in df.columns:
        df["orientation_pitch"] *= -1

    elif method == "rotate_yaw_180" and "orientation_yaw" in df.columns:
        df["orientation_yaw"] = (df["orientation_yaw"] + 180) % 360

    # (Optional) Quaternion-Manipulation — vorsichtig einsetzen
    elif method == "flip_quaternion" and all(c in df.columns for c in ["orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw"]):
        df["orientation_qx"] *= -1
        df["orientation_qy"] *= -1
        df["orientation_qz"] *= -1
        df["orientation_qw"] *= -1

    else:
        print(f"⚠️ Methode '{method}' nicht erkannt oder erforderliche Spalten fehlen.")

    return df