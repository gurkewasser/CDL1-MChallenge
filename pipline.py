#!/usr/bin/env python3
"""
full_pipeline.py

Dieses Skript führt automatisch folgende Schritte durch:

1) ZIPs entpacken und pro Session Parquet-Dateien erzeugen (in data/processed/per_file).
2) NDL_MODE=True  → Nur Trainings-Features erzeugen unter data/train/NDL/features_non_dl_train.csv
   DL_MODE=True   → Train- und Test-Sequenzen erzeugen unter:
                     data/train/DL/dl_data_train.npz
                     data/test/DL/dl_data_test.npz

Aufruf:
    python full_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

# -------------------- 1) Projekt-Root bestimmen --------------------

project_root = Path(__file__).resolve().parent
PER_FILE_DIR = project_root / "data" / "processed" / "per_file"
sys.path.append(str(project_root / "src"))

# -------------------- 2) Externe Skripte von preprocess_functions importieren --------------------
from preprocess_functions import (
    preprocess_dataframe,
    segment_dataframe,
    moving_average,
    extract_features,
    get_label_per_segment,
    process_all_zips,
)

# -------------------- 3) Konfiguration --------------------
NDL_MODE    = True   # True → nur Train-Features unter data/train/NDL/
DL_MODE     = True   # True → Train- und Test-Sequenzen unter data/train/DL/ und data/test/DL/
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Preprocessing-Parameter
DROP_INITIAL   = 8.0
DROP_FINAL     = 8.0
SAMPLING_RATE  = 50.0
SEGMENT_LENGTH = 5.0
OVERLAP        = 2.0
BUTTER_ORDER   = 4
BUTTER_CUTOFF  = 6.0
MA_WINDOW      = 0.1

# Pfade
PER_FILE_DIR = project_root / "data" / "processed" / "per_file"

# data/train/NDL, data/train/DL, data/test/DL
TRAIN_DIR_NDL = project_root  / "data" / "NDL" 
TRAIN_DIR_DL  = project_root / "data" / "DL" / "TRAIN"
TEST_DIR_DL   = project_root / "data" / "DL"  / "TEST"

# -------------------- 4) Hilfsfunktion: Verzeichnis leeren --------------------
def clear_directory(directory: Path, skip: set[str] = None):
    if not directory.exists():
        return
    for item in directory.iterdir():
        if skip and item.name in skip:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

# -------------------- 5) Kernfunktion: Preprocessing --------------------
def run_preprocessing(
    parquet_files: list[Path],
    output_dir: Path,
    dl_mode: bool,
    train_ratio: float,
    random_seed: int,
    mode: str
):
    """
    parquet_files  : Liste von Session-*.parquet-Dateien
    output_dir     : Zielordner (z.B. data/train/DL oder data/test/DL oder data/train/NDL)
    dl_mode        : False → NDL-Features, True → DL-Sequenzen
    train_ratio    : Anteil der Dateien im Trainingsset (nur bei dl_mode=True relevant)
    random_seed    : Seed für Shuffle
    mode           : 'train' oder 'test'
    """
    # 1) Zielordner anlegen (oder leeren)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(output_dir, skip=None)

    # 2) Bei DL_Mode: File-Level Split
    if dl_mode:
        np.random.seed(random_seed)
        shuffled   = np.random.permutation(parquet_files)
        split_idx  = int(len(shuffled) * train_ratio)
        train_files = list(shuffled[:split_idx])
        test_files  = list(shuffled[split_idx:])
        files = train_files if mode == "train" else test_files
    else:
        # NDL: Nur Trainingsmodus erlauben
        if mode != "train":
            print(f"  ▶ NDL_MODE=True → Überspringe run_preprocessing für mode='{mode}'.")
            return
        files = parquet_files

    if not files:
        print(f"  ⚠️ Keine Dateien für mode='{mode}', dl_mode={dl_mode}")
        return

    print(f"\n── Preprocessing: mode={mode}, dl_mode={dl_mode}, Dateien={len(files)} ──")
    all_features  = []
    all_labels    = []
    all_sequences = []
    channel_names = None
    seq_len       = int(SEGMENT_LENGTH * SAMPLING_RATE)
    total_segments = 0

    for pf in tqdm(files, desc=f"{mode.upper()} (dl={dl_mode})", ncols=100):
        try:
            raw = pd.read_parquet(pf)
        except Exception:
            continue

        # a) Trimmen + Resample
        try:
            proc = preprocess_dataframe(
                raw,
                time_col="time",
                drop_initial=DROP_INITIAL,
                drop_final=DROP_FINAL,
                sampling_rate=SAMPLING_RATE
            )
        except Exception:
            continue

        # b) Segmentieren
        try:
            segments = segment_dataframe(
                proc,
                segment_length=SEGMENT_LENGTH,
                overlap=OVERLAP
            )
        except Exception:
            continue
        if not segments:
            continue

        # c) Moving Average pro Segment
        segments_processed = []
        for seg in segments:
            try:
                seg_smooth = moving_average(
                    seg,
                    window_sec=MA_WINDOW,
                    sampling_rate=SAMPLING_RATE
                )
                segments_processed.append(seg_smooth)
            except Exception:
                segments_processed.append(seg)

        # d) Labels holen
        try:
            labels = get_label_per_segment(
                raw,
                segments_processed,
                time_col="time",
                label_col="activity"
            )
        except Exception:
            labels = [None] * len(segments_processed)

        # e1) NDL (nur Trainings-Features)
        if not dl_mode:
            try:
                feats_df = extract_features(segments_processed)
            except Exception:
                continue
            feats_df["label"] = labels
            feats_df["file"]  = pf.name
            feats_df["segment_global_id"] = np.arange(
                total_segments,
                total_segments + len(segments_processed)
            )
            feats_df = feats_df.set_index("segment_global_id")
            all_features.append(feats_df)

        # e2) DL (Sequenzen: pad/truncate)
        if dl_mode:
            if channel_names is None:
                channel_names = segments_processed[0].columns.tolist()
                seq_len = int(SEGMENT_LENGTH * SAMPLING_RATE)
            for seg in segments_processed:
                arr = np.zeros((seq_len, len(channel_names)), dtype=np.float32)
                seg_arr = np.vstack([seg[ch].values for ch in channel_names]).T
                if seg_arr.shape[0] >= seq_len:
                    arr[:] = seg_arr[:seq_len]
                else:
                    arr[:seg_arr.shape[0]] = seg_arr
                all_sequences.append(arr)

        total_segments += len(segments_processed)
        all_labels.extend(labels)

    if total_segments == 0:
        print(f"  ❌ Keine Segmente gefunden für mode='{mode}', dl_mode={dl_mode}")
        return

    # 6a) Speichern NDL-Features (nur train)
    if not dl_mode:
        df_all = pd.concat(all_features, axis=0)
        df_all.reset_index(drop=True, inplace=True)
        out_file = output_dir / "features_non_dl.csv"
        df_all.to_csv(out_file, index=False)
        print(f"✅ NDL: {len(df_all)} Segmente → {out_file}")

    # 6b) Speichern DL-Sequenzen (train/test)
    if dl_mode:
        X = np.stack(all_sequences, axis=0)
        y = np.array(all_labels, dtype=object)
        suffix = "_train" if mode == "train" else "_test"
        out_file = output_dir / f"dl_data{suffix}.npz"
        np.savez_compressed(out_file, X=X, y=y)
        print(f"✅ DL ({mode}): {X.shape[0]} Segmente → {out_file}")

# -------------------- 7) Skript-Einstiegspunkt --------------------
if __name__ == "__main__":
    # 7.1) ZIPs entpacken → Parquet pro Session
    print("\n--- Schritt 1: Entpacke ZIPs und erzeuge Parquet-Dateien (per_file) ---")
    process_all_zips()
    print("✅ Parquets liegen jetzt in data/processed/per_file/\n")

    parquet_files = list(PER_FILE_DIR.glob("*.parquet"))
    if not parquet_files:
        print("❌ Fehler: Keine .parquet-Dateien in data/processed/per_file!")
        sys.exit(1)

    # 7.2) NDL: Nur Trainings-Features erstellen unter data/train/NDL/
    print("\n--- Schritt 2: Konfiguire non Deep Learning Daten) ---")
    if NDL_MODE:
        run_preprocessing(
            parquet_files=parquet_files,
            output_dir=TRAIN_DIR_NDL,
            dl_mode=False,
            train_ratio=TRAIN_RATIO,
            random_seed=RANDOM_SEED,
            mode="train"
        )

    # 7.3) DL: Train- & Test-Sequenzen unter data/train/DL/ und data/test/DL/
    print("\n--- Schritt 3: Konfiguire Deep Learning Daten - Train und Test Split) ---")
    if DL_MODE:
        run_preprocessing(
            parquet_files=parquet_files,
            output_dir=TRAIN_DIR_DL,
            dl_mode=True,
            train_ratio=TRAIN_RATIO,
            random_seed=RANDOM_SEED,
            mode="train"
        )
        run_preprocessing(
            parquet_files=parquet_files,
            output_dir=TEST_DIR_DL,
            dl_mode=True,
            train_ratio=TRAIN_RATIO,
            random_seed=RANDOM_SEED,
            mode="test"
        )

    print("\n=== Fertig: Pipeline komplett durchgelaufen. ===")