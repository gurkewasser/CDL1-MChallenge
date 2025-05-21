import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import sys
from tqdm import tqdm

def load_and_concatenate(raw_dir: Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """
    Loads all raw data (Parquet) from a directory or a single file and returns a DataFrame.
    """
    if raw_dir.is_file():
        # If a file is provided, just read it directly
        try:
            df = pd.read_parquet(raw_dir)
            return df
        except Exception as e:
            print(f"Error reading parquet file {raw_dir}: {e}", file=sys.stderr)
            raise
    elif raw_dir.is_dir():
        files = list(raw_dir.glob(pattern))
        if not files:
            raise ValueError(f"No files matching pattern '{pattern}' found in directory {raw_dir}")
        dfs = []
        for f in tqdm(files, desc="Loading parquet files"):
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                print(f"Error reading parquet file {f}: {e}", file=sys.stderr)
                continue
        if not dfs:
            raise ValueError(f"No readable parquet files found in directory {raw_dir}")
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"{raw_dir} is neither a file nor a directory.")

def preprocess_dataframe(df: pd.DataFrame,
                         time_col: str = 'time',
                         drop_initial: float = 8.0,
                         drop_final: float = 8.0,
                         sampling_rate: float = 50.0) -> pd.DataFrame:
    """
    - Convert time column to datetime and set as index
    - Drop initial and final seconds
    - Resample numeric columns to sampling_rate (Hz)
    """
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame columns: {df.columns}")
    df[time_col] = pd.to_datetime(df[time_col], unit='ns')
    df = df.set_index(time_col).sort_index()
    if len(df) == 0:
        raise ValueError("DataFrame is empty after setting time index.")
    # Drop initial/final seconds
    start = df.index[0] + pd.Timedelta(seconds=drop_initial)
    end = df.index[-1] - pd.Timedelta(seconds=drop_final)
    df = df[(df.index >= start) & (df.index <= end)]
    if len(df) == 0:
        raise ValueError("DataFrame is empty after dropping initial/final seconds.")
    # Only numeric sensor columns
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for resampling.")
    df_num = df[num_cols]
    # Resample
    rule = f"{int(1e6 / sampling_rate)}us"
    df_resampled = df_num.resample(rule).mean().interpolate()
    if len(df_resampled) == 0:
        raise ValueError("DataFrame is empty after resampling.")
    return df_resampled

def segment_dataframe(df: pd.DataFrame,
                      segment_length: float = 5.0,
                      overlap: float = 2.0) -> list[pd.DataFrame]:
    """
    Segments DataFrame into overlapping windows (in seconds).
    """
    seg_len = pd.Timedelta(seconds=segment_length)
    ov = pd.Timedelta(seconds=overlap)
    segments = []
    t0 = df.index.min()
    while t0 + seg_len <= df.index.max():
        t1 = t0 + seg_len
        seg = df[t0:t1]
        if not seg.empty:
            segments.append(seg)
        t0 = t1 - ov
    if not segments:
        raise ValueError("No segments created from DataFrame. Check segment_length and overlap.")
    return segments

def butterworth_filter(df: pd.DataFrame,
                       order: int = 4,
                       cutoff_hz: float = 6.0,
                       sampling_rate: float = 50.0) -> pd.DataFrame:
    """
    Applies a low-pass Butterworth filter to all columns.
    """
    nyq = 0.5 * sampling_rate
    norm_cutoff = cutoff_hz / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    df_filt = df.copy()
    for col in df.columns:
        try:
            df_filt[col] = filtfilt(b, a, df[col].values)
        except Exception as e:
            print(f"Error filtering column {col}: {e}", file=sys.stderr)
            df_filt[col] = df[col]
    return df_filt

def moving_average(df: pd.DataFrame,
                   window_sec: float = 0.1,
                   sampling_rate: float = 50.0) -> pd.DataFrame:
    """
    Applies moving average to all columns.
    """
    w = int(window_sec * sampling_rate)
    if w < 1:
        w = 1
    df_smooth = df.rolling(window=w, center=True, min_periods=1).mean()
    return df_smooth

def extract_features(segments: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Computes statistical features (mean, std, min, max) per segment and column.
    Returns DataFrame with one row per segment.
    """
    rows = []
    for i, seg in enumerate(segments):
        feats = {}
        for col in seg.columns:
            feats[f"{col}_mean"] = seg[col].mean()
            feats[f"{col}_std"] = seg[col].std()
            feats[f"{col}_min"] = seg[col].min()
            feats[f"{col}_max"] = seg[col].max()
        feats['segment_id'] = i
        rows.append(feats)
    if not rows:
        raise ValueError("No features extracted: segments list is empty.")
    return pd.DataFrame(rows).set_index('segment_id')

def main(args):
    pbar_total = 8
    pbar = tqdm(total=pbar_total, desc="Preprocessing pipeline", ncols=100)
    try:
        raw = load_and_concatenate(Path(args.raw_dir))
        pbar.set_description("Loading raw data")
        pbar.update(1)
    except Exception as e:
        print(f"Failed to load data: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)
    try:
        proc = preprocess_dataframe(raw,
                                    drop_initial=args.drop_initial,
                                    drop_final=args.drop_final,
                                    sampling_rate=args.sampling_rate)
        pbar.set_description("Preprocessing dataframe")
        pbar.update(1)
    except Exception as e:
        print(f"Preprocessing failed: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)

    # Filter and smooth full signal at once
    try:
        pbar.set_description("Filtering full signal")
        proc_filt = butterworth_filter(proc,
                                        order=args.butter_order,
                                        cutoff_hz=args.butter_cutoff,
                                        sampling_rate=args.sampling_rate)
        pbar.update(1)
    except Exception as e:
        print(f"Filtering full signal failed: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)
    try:
        pbar.set_description("Smoothing full signal")
        proc_smooth = moving_average(proc_filt,
                                     window_sec=args.ma_window,
                                     sampling_rate=args.sampling_rate)
        pbar.update(1)
    except Exception as e:
        print(f"Smoothing full signal failed: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)
    # Segment the smoothed signal
    try:
        segments = segment_dataframe(proc_smooth,
                                     segment_length=args.segment_length,
                                     overlap=args.overlap)
        pbar.set_description("Segmenting dataframe")
        pbar.update(1)
    except Exception as e:
        print(f"Segmentation failed: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)
    # Extract features for non-DL models
    try:
        features_df = extract_features(segments)
        pbar.set_description("Extracting features")
        pbar.update(1)
    except Exception as e:
        print(f"Feature extraction failed: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)
    # Build raw sequences for DL models
    try:
        raw_seqs = []
        for i, seg in enumerate(segments):
            row = {'segment_id': i}
            for col in seg.columns:
                row[col] = seg[col].tolist()
            raw_seqs.append(row)
        sequences_df = pd.DataFrame(raw_seqs).set_index('segment_id')
        pbar.set_description("Building sequences")
        pbar.update(1)
    except Exception as e:
        print(f"Sequence building failed: {e}", file=sys.stderr)
        pbar.close()
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out / 'features.parquet')
    sequences_df.to_parquet(out / 'sequences.parquet')
    pbar.set_description("Saving outputs")
    pbar.update(1)
    pbar.close()
    print(f"Saved {len(features_df)} segments: features & sequences to {out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocessing Pipeline nach Notebook-Schritten')
    p.add_argument('--raw_dir', type=str, required=True, help='Ordner mit Roh-Parquet-Dateien oder Einzeldatei')
    p.add_argument('--output_dir', type=str, required=True, help='Zielordner für Ausgaben')
    p.add_argument('--drop_initial', type=float, default=8.0, help='Sekunden am Anfang droppen')
    p.add_argument('--drop_final', type=float, default=8.0, help='Sekunden am Ende droppen')
    p.add_argument('--sampling_rate', type=float, default=50.0, help='Resample-Rate in Hz')
    p.add_argument('--segment_length', type=float, default=5.0, help='Segment-Länge in Sekunden')
    p.add_argument('--overlap', type=float, default=2.0, help='Overlap in Sekunden')
    p.add_argument('--butter_order', type=int, default=4, help='Butterworth Filter Order')
    p.add_argument('--butter_cutoff', type=float, default=6.0, help='Butterworth Cutoff Hz')
    p.add_argument('--ma_window', type=float, default=0.1, help='Moving Average Window in Sek.')
    args = p.parse_args()
    main(args)