
"""
Funktionen
----------
1. **ZIP‑Entpacken**:  Jede ZIP‑Datei wird in eine Parquet‑Datei pro Session unter
   `data/processed/per_file/` konvertiert (Step 1).
2. **Non‑Deep‑Learning (NDL)**:   Für **Train** und **Test** werden Feature‑CSVs erzeugt, die sich
   **nicht mehr überschreiben**.  Dafür legt das Skript getrennte Ordner an:    
   * `data/NDL/TRAIN/features_non_dl.csv`    
   * `data/NDL/TEST/features_non_dl.csv`
3. **Deep Learning (DL)**:  Für **Train**, **Test** werden
   Sequenz‑NPZ‑Dateien erzeugt:    
   * `data/DL/TRAIN/dl_data_train.npz`    
   * `data/DL/TEST/dl_data_test.npz`    

Start
-----
    python full_pipeline.py
"""

from __future__ import annotations

import sys
import shutil
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1) Projekt‑Root + Helfer‑Importe
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent  # Basisverzeichnis des Projekts
PER_FILE_DIR = project_root / "data" / "processed" / "per_file"  # Zielordner Parquets

# src‑Ordner (enthält preprocess_functions.py) zum Python‑Pfad hinzufügen
sys.path.append(str(project_root / "src"))

# ---- Externe Preprocessing‑Funktionen ----------------------------------------
from preprocess_functions import (
    preprocess_dataframe,   # zeitliches Trimmen + Resampling
    segment_dataframe,      # Sliding‑Window‑Segmentierung
    moving_average,         # optionale Glättung je Segment
    extract_features,       # Feature‑Engineering (Statistiken etc.)
    get_label_per_segment,  # Label pro Segment bestimmen
    process_all_zips,       # ZIP -> Parquet‑Konverter
)

# -----------------------------------------------------------------------------
# 2) Konfiguration
# -----------------------------------------------------------------------------

NDL_MODE = True  # klassische ML‑Features erzeugen
DL_MODE  = True  # DL‑Sequenzen erzeugen

# Split‑Verhältnis
TRAIN_RATIO = 0.75       # 75% Train, 25% Test
RANDOM_SEED = 690       # für reproduzierbares Shuffle

# Basis‑Preprocessing‑Parameter
DROP_INITIAL   = 8.0     # Sekunden am Anfang jedes Mitschnitts verwerfen
DROP_FINAL     = 8.0     # Sekunden am Ende verwerfen
SAMPLING_RATE  = 50.0    # Ziel‑Sampling‑Rate in Hz
SEGMENT_LENGTH = 5.0     # Segmentlänge in Sek.
OVERLAP        = 2.5    # Überlappung in Sek.
MA_WINDOW      = 0.1     # Moving‑Average‑Fenster in Sek.

# -----------------------------------------------------------------------------
# 3) Ziel‑Verzeichnisse
# -----------------------------------------------------------------------------
# Parquets liegen in  data/processed/per_file/  (siehe oben)

# NDL
NDL_TRAIN_DIR = project_root / "data" / "NDL" / "TRAIN"
NDL_TEST_DIR  = project_root / "data" / "NDL" / "TEST"

# DL 
TRAIN_DIR_DL = project_root / "data" / "DL" / "TRAIN"
TEST_DIR_DL  = project_root / "data" / "DL" / "TEST"

# -----------------------------------------------------------------------------
# 4) Hilfsfunktionen
# -----------------------------------------------------------------------------

def get_file_activities(pf: Path) -> set[str]:
    """Liest nur die Spalte 'activity' und liefert das Set vorhandener Labels."""
    try:
        return set(pd.read_parquet(pf, columns=["activity"])["activity"].dropna().unique())
    except Exception:
        return set()


def count_segments_by_activity(pf: Path) -> dict[str, int]:
    """Zählt erwartete Segmentanzahl pro Aktivität in einer Datei."""
    try:
        raw = pd.read_parquet(pf)
        proc = preprocess_dataframe(
            raw,
            time_col="time",
            drop_initial=DROP_INITIAL,
            drop_final=DROP_FINAL,
            sampling_rate=SAMPLING_RATE,
        )
        segments = segment_dataframe(proc, SEGMENT_LENGTH, OVERLAP)
        if not segments:
            return {}
        segments_proc = [
            moving_average(seg, MA_WINDOW, SAMPLING_RATE)
            for seg in segments
        ]
        labels = get_label_per_segment(
            raw, segments_proc, time_col="time", label_col="activity"
        )
        counts = {}
        for label in labels:
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        return counts
    except Exception:
        return {}

def create_stratified_split(
    parquet_files: list[Path],
    train_ratio: float,
    random_seed: int,
    max_attempts: int = 1000,
) -> dict[str, list[Path]]:
    """
    Erstellt einen Split in Trainings- und Testdaten auf Dateiebene, wobei die
    Verteilung der Aktivitätsklassen möglichst ähnlich ist. Kein Mindestsegment-Kriterium.
    """
    rng = np.random.default_rng(random_seed)

    # Datei → Anzahl Segmente pro Label
    file2segcounts = {}
    all_labels = set()
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=["activity"])
            counts = df["activity"].value_counts().to_dict()
            file2segcounts[pf] = {str(k): int(v) for k, v in counts.items()}
            all_labels.update(counts.keys())
        except Exception:
            file2segcounts[pf] = {}

    # Bewertungsfunktion: maximale absolute Abweichung (%) je Klasse
    def imbalance(split):
        def count(files):
            total = {}
            for f in files:
                for label, n in file2segcounts.get(f, {}).items():
                    total[label] = total.get(label, 0) + n
            return total

        train_counts = count(split["train"])
        test_counts = count(split["test"])
        total_counts = {k: train_counts.get(k, 0) + test_counts.get(k, 0) for k in all_labels}

        max_imbalance = 0.0
        for label in all_labels:
            t = train_counts.get(label, 0)
            test = test_counts.get(label, 0)
            total = total_counts[label]
            if total > 0:
                pct_train = t / total
                max_imbalance = max(max_imbalance, abs(pct_train - train_ratio))
        return max_imbalance

    # Split mit minimaler Imbalance
    best_split = None
    best_score = float("inf")
    for _ in range(max_attempts):
        shuffled = rng.permutation(parquet_files)
        n_total = shuffled.size
        n_train = int(n_total * train_ratio)

        split = {
            "train": list(shuffled[:n_train]),
            "test": list(shuffled[n_train:]),
        }

        score = imbalance(split)
        if score < best_score:
            best_score = score
            best_split = split
            if best_score == 0:
                break

    if best_split is None:
        print("❌ Kein gültiger Split gefunden.")
        return {"train": [], "test": []}

    print(f"✅ Split gewählt mit maximaler Label-Imbalance von {best_score:.2%}")
    return best_split

def clear_directory(directory: Path, *, skip: set[str] | None = None) -> None:
    """Löscht alles im Verzeichnis (außer optional *skip*)."""
    if not directory.exists():
        return
    for item in directory.iterdir():
        if skip and item.name in skip:
            continue
        (shutil.rmtree if item.is_dir() else Path.unlink)(item)

# -----------------------------------------------------------------------------
# 5) Kerneinheit:  run_preprocessing()
# -----------------------------------------------------------------------------

def run_preprocessing(
    *,
    files: list[Path],
    output_dir: Path,
    dl_mode: bool,
) -> None:
    """Verarbeitet eine Liste Parquets und erzeugt Split‑Artefakte (Features oder Sequenzen)."""

    if not files:
        print(f"⚠️  Keine Dateien übergeben (dl_mode={dl_mode})")
        return

    # ---------------------------------------------------------------------
    # 5‑A) Zielordner vorbereiten
    #      Bei jedem Aufruf wird nur *sein* Ordner geleert → keine Überschreibungen.
    # ---------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir)  # leert nur den eigenen Split‑Ordner


    # ---------------------------------------------------------------------
    # 5‑B) Initialisierung von Sammlern
    # ---------------------------------------------------------------------
    print(f"\n── Preprocessing  dl_mode={dl_mode}  files={len(files)} ──")

    all_features: list[pd.DataFrame] = []   # Feature‑DataFrames je Segment
    all_sequences: list[np.ndarray]   = []  # DL‑Sequenzen
    all_labels:    list[str | None]   = []  # Labels (können None sein)

    channel_names: list[str] | None = None  # Merkmalskanäle für DL
    seq_len = int(SEGMENT_LENGTH * SAMPLING_RATE)
    global_seg_counter = 0                 # laufender Index über alle Segmente

    # ---------------------------------------------------------------------
    # 5‑C) Haupt‑Loop über alle Dateien des Splits
    # ---------------------------------------------------------------------
    for pf in tqdm(files, desc="PREPROCESS", ncols=90):
        # (1) Parquet lesen
        try:
            raw = pd.read_parquet(pf)
        except Exception:
            continue  # Datei fehlerhaft → überspringen

        # (2) Trimmen + Resampling
        try:
            proc = preprocess_dataframe(
                raw,
                time_col="time",
                drop_initial=DROP_INITIAL,
                drop_final=DROP_FINAL,
                sampling_rate=SAMPLING_RATE,
            )
        except Exception:
            continue

        # (3) Sliding‑Window‑Segmentierung
        try:
            segments = segment_dataframe(proc, SEGMENT_LENGTH, OVERLAP)
        except Exception:
            continue
        if not segments:
            continue

        # (4) Optionale Moving‑Average‑Glättung
        segments_proc: list[pd.DataFrame] = []
        for seg in segments:
            try:
                seg_sm = moving_average(seg, MA_WINDOW, SAMPLING_RATE)
                segments_proc.append(seg_sm)
            except Exception:
                segments_proc.append(seg)  # Fallback: ungesmoothed

        # (5) Label pro Segment
        try:
            labels = get_label_per_segment(
                raw, segments_proc, time_col="time", label_col="activity"
            )
        except Exception:
            labels = [None] * len(segments_proc)

        # (6a) **NDL‑Pfad**: Feature‑Engineering
        if not dl_mode:
            try:
                feats = extract_features(segments_proc)  # DataFrame pro Datei
            except Exception:
                continue
            feats["label"] = labels
            feats["file"]  = pf.name
            feats["segment_global_id"] = np.arange(
                global_seg_counter, global_seg_counter + len(segments_proc)
            )
            feats = feats.set_index("segment_global_id")
            all_features.append(feats)

        # (6b) **DL‑Pfad**: Sequenzen padd/tuncate → ndarray
        if dl_mode:
            if channel_names is None:
                channel_names = segments_proc[0].columns.tolist()
            for seg in segments_proc:
                arr = np.zeros((seq_len, len(channel_names)), dtype=np.float32)  # Padding
                seg_arr = seg[channel_names].to_numpy()
                arr[: min(seq_len, seg_arr.shape[0])] = seg_arr[:seq_len]
                all_sequences.append(arr)

        global_seg_counter += len(segments_proc)
        all_labels.extend(labels)

    # ---------------------------------------------------------------------
    # 5‑D) Persistieren der Artefakte
    # ---------------------------------------------------------------------
    if global_seg_counter == 0:
        print(f"❌  Keine Segmente für dl_mode={dl_mode}")
        return
    
    if not dl_mode:
        df_all = pd.concat(all_features).reset_index(drop=True)
        suffix = output_dir.name.lower()
        out_csv = output_dir / f"features_non_dl_{suffix}.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"✅ NDL: {len(df_all)} Segmente → {out_csv}")

    else:
        X = np.stack(all_sequences)
        y = np.asarray(all_labels, dtype=object)
        suffix = output_dir.name.lower()
        out_npz = output_dir / f"dl_data_{suffix}.npz"
        np.savez_compressed(out_npz, X=X, y=y)
        print(f"✅ DL: {X.shape[0]} Segmente → {out_npz}")

# -----------------------------------------------------------------------------
# 6) Skript‑Einstiegspunkt
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 0) Erstelle die Ordner raw, processed, unpacked falls nicht vorhanden
    # -----------------------------------------------------------------
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    unpacked_dir = data_dir / "unpacked"

    for d in [raw_dir, processed_dir, unpacked_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zip_pattern = re.compile(r".+_data_raw_iseni_hatemo\.zip$")
    found_zip = None
    for f in data_dir.glob("*_data_raw_iseni_hatemo.zip"):
        if zip_pattern.match(f.name):
            found_zip = f
            break

    if found_zip is not None:
        new_zip_name = "cdl1_data_raw_iseni_hatemo.zip"
        renamed_zip_path = data_dir / new_zip_name

        try:
            # Umbenennen (falls nötig)
            if found_zip.name != new_zip_name:
                found_zip.rename(renamed_zip_path)
                print(f"✅ Umbenannt: {found_zip} → {renamed_zip_path}")
            else:
                renamed_zip_path = found_zip  # Bereits richtiger Name

            # Nach data/raw verschieben
            zip_dst = raw_dir / new_zip_name
            shutil.move(str(renamed_zip_path), str(zip_dst))
            print(f"✅ Verschoben: {renamed_zip_path} → {zip_dst}")
        except Exception as e:
            print(f"⚠️  Konnte ZIP nicht umbenennen/verschieben: {e}")
    else:
        print(f"⚠️  Keine passende ZIP-Datei gefunden im {data_dir} (erwartet: *_data_raw_iseni_hatemo.zip)")

    # -----------------------------------------------------------------
    # 7) Schritt 1: ZIPs entpacken und Parquet‑Dateien erzeugen
    # -----------------------------------------------------------------
    print("\n--- Schritt 1: Entpacke ZIP-Dateien & erstelle Parquets ---")
    process_all_zips()
    print("✅ Parquets unter data/processed/per_file/ vorhanden.\n")

    parquet_files = list(PER_FILE_DIR.glob("*.parquet"))
    if not parquet_files:
        sys.exit("❌  Keine .parquet-Dateien gefunden!")

    # --- Globaler, label‑abgedeckter Split ----------------------------------
    split_dict = create_stratified_split(
        parquet_files, TRAIN_RATIO, RANDOM_SEED
    )

    # -----------------------------------------------------------------
    # 8) Non‑Deep‑Learning: nur Train & Test (getrennte Ordner)
    # -----------------------------------------------------------------
    if NDL_MODE:
        print("\n--- Schritt 2: Verarbeite NDL (Train & Test) ---")
        # Train‑Features
        run_preprocessing(
            files=split_dict["train"],
            output_dir=NDL_TRAIN_DIR,
            dl_mode=False,
        )
        # Test‑Features
        run_preprocessing(
            files=split_dict["test"],
            output_dir=NDL_TEST_DIR,
            dl_mode=False,
        )

    # -----------------------------------------------------------------
    # 9) Deep Learning: Train / Test
    # -----------------------------------------------------------------
    if DL_MODE:
        print("\n--- Schritt 3: Verarbeite DL (Train & Test ) ---")
        # Train‑Sequenzen
        run_preprocessing(
            files=split_dict["train"],
            output_dir=TRAIN_DIR_DL,
            dl_mode=True,
        )
        # Test‑Sequenzen
        run_preprocessing(
            files=split_dict["test"],
            output_dir=TEST_DIR_DL,
            dl_mode=True,
        )

    # -----------------------------------------------------------------
    # 10) Fertig
    # -----------------------------------------------------------------
    print("\n=== Pipeline abgeschlossen. ===")
