import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
from tqdm import tqdm

import rootutils
from matplotlib import pyplot as plt


# ---------------------------------------------------------
# Projekt-Root und Verzeichnis-Setup (plattformunabhängig)
# ---------------------------------------------------------
try:
    # Versuche, rootutils zu verwenden, falls verfügbar
    ROOT = rootutils.setup_root(__file__, pythonpath=True, cwd=True)
    ROOT = Path(ROOT)
except Exception:
    # Fallback: zwei Ebenen über dieser Datei
    ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# Projekt-Root und Verzeichnis-Setup
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
USER_DATA    = ROOT / "data"
INPUT_DIR    = USER_DATA / "raw"
UNPACK_DIR   = USER_DATA / "unpacked"
OUTPUT_DIR   = USER_DATA / "processed"
PER_FILE_DIR = OUTPUT_DIR / "per_file"
OUTPUT_PATH  = OUTPUT_DIR / "raw_data.parquet"

# Erstelle nötige Verzeichnisse
for d in [USER_DATA, INPUT_DIR, UNPACK_DIR, OUTPUT_DIR, PER_FILE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Definition relevanter Sensor-Dateien und Metadaten
# ---------------------------------------------------------
SENSOR_FILES = {
    "Accelerometer.csv": ["x", "y", "z"],
    "Magnetometer.csv": ["x", "y", "z"],
    "Gravity.csv": ["x", "y", "z"],
    "Gyroscope.csv": ["x", "y", "z"],
    "Orientation.csv": ["pitch", "roll", "yaw", "qw", "qx", "qy", "qz"],
}

METADATA_FILE = "Metadata.csv"


# ---------------------------------------------------------
# Funktionen zur Entpackung
# ---------------------------------------------------------
def extract_entire_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Entpackt die gegebene ZIP-Datei vollständig in extract_to/zip_path.stem.
    """
    extract_dir = extract_to / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def extract_nested_zip_files(parent_dir: Path) -> None:
    """
    Sucht in parent_dir nach *.zip und entpackt sie in UNPACK_DIR/<zipname>.
    Funktioniert nur eine Ebene tief – für tiefer verschachtelte ZIPs sollte
    process_all_zips() den rekursiven Extraktor verwenden.
    """
    for sub_zip in parent_dir.glob("*.zip"):
        target_folder = UNPACK_DIR / sub_zip.stem
        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(sub_zip, "r") as zip_ref:
                zip_ref.extractall(target_folder)


def extract_all_nested_zips(zip_path: Path, extract_to: Path) -> None:
    """
    Entpackt zip_path in extract_to/zip_path.stem und
    iterativ alle darin enthaltenen verschachtelten ZIP-Dateien.
    Jeder verschachtelte ZIP wird nach dem Entpacken gelöscht,
    sodass keine Endlosschleife entsteht und BadZipFile-Fälle
    abgefangen werden.
    """
    # 1) Erstes Entpacken in einen eigenen Unterordner:
    extract_dir = extract_to / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        # ZIP ist defekt oder kein ZIP → überspringen
        return

    # 2) Alle entdeckten ZIP-Dateien in dieser Schicht und Unterordnern einsammeln
    zips_to_process: list[Path] = []
    for nested in extract_dir.rglob("*.zip"):
        zips_to_process.append(nested)

    # 3) Iteratives Abarbeiten (hier LIFO)
    while zips_to_process:
        current_zip = zips_to_process.pop()
        target_dir = current_zip.parent / current_zip.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(current_zip, "r") as nested_zf:
                nested_zf.extractall(target_dir)
        except zipfile.BadZipFile:
            # Datei ist kein valides ZIP → löschen und überspringen
            current_zip.unlink(missing_ok=True)
            continue

        # Nach dem Entpacken die verschachtelte ZIP löschen
        current_zip.unlink(missing_ok=True)

        # Suche im neu angelegten Ordner nach weiteren verschachtelten ZIPs
        for deeper_zip in target_dir.rglob("*.zip"):
            zips_to_process.append(deeper_zip)


# ---------------------------------------------------------
# Funktionen zum Einlesen der Rohdaten
# ---------------------------------------------------------
def get_label_from_filename(filename: str) -> str:
    """
    Extrahiert das Label vor dem ersten Unterstrich im Ordner-/Dateinamen.
    """
    return filename.split("_")[0]


def load_metadata(folder: Path) -> dict[str, str]:
    """
    Liest aus METADATA_FILE innerhalb von folder die Spalten 'device id',
    'device name' und 'appVersion' (falls vorhanden). Gibt ein Dictionary zurück.
    """
    meta_path = folder / METADATA_FILE
    if not meta_path.exists():
        return {"device_id": None, "device_name": None, "app_version": None}

    df = pd.read_csv(meta_path)
    return {
        "device_id": df.get("device id", [None])[0],
        "device_name": df.get("device name", [None])[0],
        "app_version": df.get("appVersion", [None])[0],
    }


def load_sensor_data(folder: Path) -> pd.DataFrame | None:
    """
    Liest alle Sensor-Dateien aus SENSOR_FILES in folder ein (falls vorhanden),
    konvertiert 'time' in datetime, benennt die Spalten um und führt per
    merge_asof eine zeitbasierte Zusammenführung durch. Gibt None zurück, wenn
    keine der erwarteten CSVs existiert.
    """
    dfs: list[pd.DataFrame] = []
    time_column: pd.DataFrame | None = None

    for fname, cols in SENSOR_FILES.items():
        fpath = folder / fname
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)
        if "seconds_elapsed" not in df.columns or "time" not in df.columns:
            continue

        # 'time' in datetime konvertieren
        df["time"] = pd.to_datetime(df["time"], unit="ns")

        if time_column is None:
            time_column = df[["seconds_elapsed", "time"]].copy()

        # Nur numerische Sensor-Spalten behalten
        sub_df = df[["seconds_elapsed"] + cols].copy()

        # Spalten umbenennen: Accelerometer.csv → accelerometer_x, accelerometer_y, …
        base = fname.replace(".csv", "").lower()
        for col in cols:
            sub_df.rename(columns={col: f"{base}_{col}"}, inplace=True)

        dfs.append(sub_df)

    if not dfs:
        return None

    # Merge aller Sensor-DataFrames nach seconds_elapsed
    merged = dfs[0].sort_values("seconds_elapsed")
    for d in dfs[1:]:
        merged = pd.merge_asof(merged, d.sort_values("seconds_elapsed"), on="seconds_elapsed")

    # Zum Schluss die time-Spalte aus time_column hinzufügen
    merged = pd.merge_asof(merged.sort_values("seconds_elapsed"),
                           time_column.sort_values("seconds_elapsed"),
                           on="seconds_elapsed")
    return merged


def remove_macosx_dir(unpack_dir: Path) -> None:
    """
    Entfernt __MACOSX-Ordner, falls er in unpack_dir vorhanden ist.
    """
    macosx_dir = unpack_dir / "__MACOSX"
    if macosx_dir.exists() and macosx_dir.is_dir():
        shutil.rmtree(macosx_dir)


# ---------------------------------------------------------
# Prozess: Alle ZIPs entpacken, CSVs einlesen und Parquet schreiben
# ---------------------------------------------------------
def process_all_zips() -> None:
    """
    1) Entpackt alle ZIP-Dateien aus INPUT_DIR.
    2) Entfernt __MACOSX, liest pro Session die CSVs ein,
       schreibt pro-Session Parquet in PER_FILE_DIR und
       eine Gesamttabelle in OUTPUT_PATH.
    """
    print("🔍 Suche und entpacke alle .zip Dateien …")
    zip_paths = list(INPUT_DIR.glob("*.zip"))
    if not zip_paths:
        print("⚠️  Keine ZIP-Dateien in INPUT_DIR gefunden.")
        return

    # 1) Zuerst alle ZIPs entpacken
    for zip_path in tqdm(zip_paths, desc="Entpacke ZIPs"):
        extract_all_nested_zips(zip_path, UNPACK_DIR)

    # 2) __MACOSX-Ordner entfernen (falls vorhanden)
    remove_macosx_dir(UNPACK_DIR)

    raw_rows: list[pd.DataFrame] = []
    zip_path_lookup = {f.stem: f for f in zip_paths}
    PER_FILE_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Alle Unterordner finden, die tatsächlich Sensor-CSV-Dateien enthalten
    session_dirs: set[Path] = set()
    for csv_path in UNPACK_DIR.rglob("*.csv"):
        if "__MACOSX" in csv_path.parts:
            continue
        session_dirs.add(csv_path.parent)

    if not session_dirs:
        print("⚠️ Keine Sensor-CSV-Dateien gefunden → überspringe Verarbeitung.")
    else:
        for folder in tqdm(sorted(session_dirs), desc="Processing sessions"):
            session_name = folder.name
            label = get_label_from_filename(session_name)

            # MD5-Hash aus allen Session-CSV-Dateien berechnen
            hash_md5 = hashlib.md5()
            for csv_file in sorted(folder.rglob("*.csv")):
                with open(csv_file, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
            file_hash = hash_md5.hexdigest()

            # Metadata- und Sensor-Daten laden
            metadata  = load_metadata(folder)
            sensor_df = load_sensor_data(folder)
            if sensor_df is None:
                continue

            sensor_df["activity"]    = label
            sensor_df["device_id"]   = metadata["device_id"]
            sensor_df["device_name"] = metadata["device_name"]
            sensor_df["app_version"] = metadata["app_version"]
            sensor_df["file_hash"]   = file_hash

            raw_rows.append(sensor_df)

            # Pro-Session Parquet speichern
            per_file_name = f"{session_name}.parquet"
            per_file_path = PER_FILE_DIR / per_file_name
            sensor_df.to_parquet(per_file_path, index=False)

    # 4) Gesamtes DataFrame zusammenführen und abspeichern
    if raw_rows:
        df_all = pd.concat(raw_rows, ignore_index=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(OUTPUT_PATH, index=False)
        print(f"✅ Rohdaten (komplett) gespeichert nach: {OUTPUT_PATH}")
        print(
            f"✅ {len(raw_rows)} Per-Session Parquet-Dateien gespeichert in: {PER_FILE_DIR}"
        )
    else:
        print("⚠️ Keine gültigen Daten gefunden; kein Parquet erstellt.")

# ---------------------------------------------------------
# Bestehende Preprocessing-Funktionen beibehalten
# ---------------------------------------------------------
def preprocess_dataframe(
    df: pd.DataFrame,
    time_col: str = "time",
    drop_initial: float = 8.0,
    drop_final: float = 8.0,
    sampling_rate: float = 50.0,
) -> pd.DataFrame:
    """
    1. Konvertiere 'time' (Nanosekunden) in pandas datetime.
    2. Setze den Zeitstempel als Index und sortiere.
    3. Entferne die ersten/letzten Sekunden (Randeffekte).
    4. Resample auf 'sampling_rate' (Hz) mit Mittelwert + linearer Interpolation.
    """
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(
            f"Zeitspalte '{time_col}' nicht gefunden. Verfügbare Spalten: {df.columns.tolist()}"
        )

    # Zeitkonvertierung: Nanosekunden → pandas datetime
    df[time_col] = pd.to_datetime(df[time_col], unit="ns")
    df = df.set_index(time_col).sort_index()
    if df.empty:
        raise ValueError("DataFrame ist nach Setzen des Zeit-Index leer.")

    # Trimmen der Randzeiten
    start = df.index[0] + pd.Timedelta(seconds=drop_initial)
    end = df.index[-1] - pd.Timedelta(seconds=drop_final)
    df = df[(df.index >= start) & (df.index <= end)]
    if df.empty:
        raise ValueError("DataFrame ist nach Entfernen der Randzeiten leer.")

    # Nur numerische Spalten für Resampling auswählen
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError("Keine numerischen Spalten für das Resampling gefunden.")
    df_num = df[num_cols]

    # Resample-Regel in Mikrosekunden (z. B. '20000us' für 50 Hz)
    rule = f"{int(1e6 / sampling_rate)}us"
    df_resampled = df_num.resample(rule).mean().interpolate()
    if df_resampled.empty:
        raise ValueError("DataFrame ist nach Resampling leer.")

    return df_resampled


def segment_dataframe(
    df: pd.DataFrame,
    segment_length: float = 5.0,
    overlap: float = 2.0,
) -> list[pd.DataFrame]:
    """
    Teilt DataFrame in überlappende Fenster auf:
    - segment_length: Länge in Sekunden (z. B. 5.0)
    - overlap: Überlappung in Sekunden (z. B. 2.0)
    """
    seg_len = pd.Timedelta(seconds=segment_length)
    ov = pd.Timedelta(seconds=overlap)
    segments: list[pd.DataFrame] = []

    t0 = df.index.min()
    t_max = df.index.max()
    while t0 + seg_len <= t_max:
        t1 = t0 + seg_len
        seg = df[t0:t1]
        if not seg.empty:
            segments.append(seg)
        t0 = t1 - ov

    if not segments:
        raise ValueError("Keine Segmente erstellt. Überprüfe 'segment_length' und 'overlap'.")
    return segments


def moving_average(
    df: pd.DataFrame, window_sec: float = 0.1, sampling_rate: float = 50.0
) -> pd.DataFrame:
    """
    Wendet einen gleitenden Durchschnitt an:
    - window_sec: Fenstergröße in Sekunden (z. B. 0.1)
    - sampling_rate: Sampling-Rate in Hz (z. B. 50)
    """
    w = int(window_sec * sampling_rate)
    if w < 1:
        w = 1
    return df.rolling(window=w, center=True, min_periods=1).mean()


def extract_features(segments: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Berechnet pro Segment:
    - Mittelwert, Std, Min, Max jeder Spalte.
    Gibt DataFrame zurück mit einer Zeile pro Segment.
    """
    rows: list[dict[str, float]] = []
    for i, seg in enumerate(segments):
        feats: dict[str, float] = {}
        for col in seg.columns:
            feats[f"{col}_mean"] = seg[col].mean()
            feats[f"{col}_std"] = seg[col].std()
            feats[f"{col}_min"] = seg[col].min()
            feats[f"{col}_max"] = seg[col].max()
        feats["segment_id"] = i
        rows.append(feats)

    if not rows:
        raise ValueError("Keine Segmente vorhanden. Feature-Extraktion nicht möglich.")
    return pd.DataFrame(rows).set_index("segment_id")


def get_label_per_segment(
    raw_df: pd.DataFrame,
    segments: list[pd.DataFrame],
    time_col: str = "time",
    label_col: str = "activity",
) -> list:
    """
    Ordnet jedem Segment das am häufigsten vorkommende Label aus raw_df zu.
    Gibt Liste der Labels (Modalwert) zurück; None bei fehlenden Labels.
    """
    if label_col not in raw_df.columns:
        raise ValueError(f"Spalte '{label_col}' nicht in raw_df vorhanden.")

    # Get the most common label from the entire raw_df
    most_common_label = raw_df[label_col].mode().iloc[0]
    
    # If we have a single label for the entire file, use that for all segments
    if len(raw_df[label_col].unique()) == 1:
        return [most_common_label] * len(segments)
    
    # Otherwise, try to get labels per segment
    labels: list = []
    for seg in segments:
        start_ts = seg.index.min()
        end_ts = seg.index.max()
        
        # Get all labels that fall within this time window
        mask = (raw_df[time_col] >= start_ts) & (raw_df[time_col] <= end_ts)
        seg_labels = raw_df.loc[mask, label_col]
        
        if not seg_labels.empty:
            labels.append(seg_labels.mode().iloc[0])
        else:
            # If no labels found in this segment, use the most common label from the file
            labels.append(most_common_label)
    
    return labels


def get_files_list(raw_dir: Path, pattern: str = "*.parquet") -> list[Path]:
    """
    Gibt eine Liste von Parquet-Dateien zurück:
    - Wenn raw_dir eine Datei ist, zurück als einzelne Liste.
    - Wenn raw_dir ein Verzeichnis ist, alle Dateien, die zum Pattern passen.
    """
    if raw_dir.is_file():
        return [raw_dir]
    elif raw_dir.is_dir():
        files = list(raw_dir.glob(pattern))
        if not files:
            raise ValueError(f"Keine Dateien mit Muster '{pattern}' in '{raw_dir}' gefunden.")
        return files
    else:
        raise ValueError(f"'{raw_dir}' ist weder eine Datei noch ein Verzeichnis.")
