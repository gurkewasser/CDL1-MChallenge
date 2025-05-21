import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import sys
from tqdm import tqdm
import shutil

def clear_directory(directory):
    """
    Delete all contents of a directory, but not the directory itself.
    """
    if not directory.exists():
        return
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

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

def get_label_per_segment(raw_df, segments, time_col='time', label_col='activity'):
    """
    For each segment, get the most frequent label in the corresponding time window.
    """
    if label_col not in raw_df.columns:
        raise ValueError(f"No '{label_col}' column in raw data for labels")
    label_series = raw_df.set_index(pd.to_datetime(raw_df[time_col], unit='ns'))[label_col]
    labels = []
    for seg in segments:
        seg_label = label_series[seg.index.min():seg.index.max()]
        labels.append(seg_label.mode()[0] if not seg_label.empty else None)
    return labels

def get_files_list(raw_dir: Path, pattern: str = "*.parquet"):
    if raw_dir.is_file():
        return [raw_dir]
    elif raw_dir.is_dir():
        files = list(raw_dir.glob(pattern))
        if not files:
            raise ValueError(f"No files matching pattern '{pattern}' found in directory {raw_dir}")
        return files
    else:
        raise ValueError(f"{raw_dir} is neither a file nor a directory.")

def main(args):
    raw_dir = Path(args.raw_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Clear the output directory before writing new files
    clear_directory(out)

    files = get_files_list(raw_dir)
    all_features = []
    all_labels = []
    all_sequences = []
    channel_names = None
    seq_len = int(args.segment_length * args.sampling_rate)
    total_segments = 0

    pbar = tqdm(files, desc="Processing files", ncols=100)
    for file_idx, file in enumerate(pbar):
        pbar.set_description(f"Processing {file.name}")
        try:
            raw = pd.read_parquet(file)
        except Exception as e:
            print(f"Error reading parquet file {file}: {e}", file=sys.stderr)
            continue

        try:
            proc = preprocess_dataframe(raw,
                                       drop_initial=args.drop_initial,
                                       drop_final=args.drop_final,
                                       sampling_rate=args.sampling_rate)
        except Exception as e:
            print(f"Preprocessing failed for {file}: {e}", file=sys.stderr)
            continue

        try:
            proc_filt = butterworth_filter(proc,
                                           order=args.butter_order,
                                           cutoff_hz=args.butter_cutoff,
                                           sampling_rate=args.sampling_rate)
        except Exception as e:
            print(f"Filtering failed for {file}: {e}", file=sys.stderr)
            continue

        try:
            proc_smooth = moving_average(proc_filt,
                                         window_sec=args.ma_window,
                                         sampling_rate=args.sampling_rate)
        except Exception as e:
            print(f"Smoothing failed for {file}: {e}", file=sys.stderr)
            continue

        try:
            segments = segment_dataframe(proc_smooth,
                                         segment_length=args.segment_length,
                                         overlap=args.overlap)
        except Exception as e:
            print(f"Segmentation failed for {file}: {e}", file=sys.stderr)
            continue

        if not segments:
            continue

        try:
            features_df = extract_features(segments)
        except Exception as e:
            print(f"Feature extraction failed for {file}: {e}", file=sys.stderr)
            continue

        # Get labels for each segment
        try:
            labels = get_label_per_segment(raw, segments, time_col='time', label_col='activity')
        except Exception as e:
            print(f"Label extraction failed for {file}: {e}", file=sys.stderr)
            labels = [None] * len(segments)

        features_df['label'] = labels
        features_df['file'] = file.name
        features_df['segment_global_id'] = np.arange(total_segments, total_segments + len(segments))
        features_df = features_df.set_index('segment_global_id')
        all_features.append(features_df)
        all_labels.extend(labels)

        # For DL: build sequences
        if channel_names is None:
            channel_names = segments[0].columns.tolist()
        for seg in segments:
            # Pad or trim to seq_len
            arr = np.zeros((seq_len, len(channel_names)), dtype=np.float32)
            seg_arr = np.vstack([seg[ch].values for ch in channel_names]).T
            if seg_arr.shape[0] >= seq_len:
                arr[:] = seg_arr[:seq_len]
            else:
                arr[:seg_arr.shape[0]] = seg_arr
            all_sequences.append(arr)
        total_segments += len(segments)

    # After all files: concatenate and save
    if not all_features or not all_sequences:
        print("No valid segments found in any file.", file=sys.stderr)
        sys.exit(1)

    features_all_df = pd.concat(all_features, axis=0)
    features_all_df.reset_index(drop=True, inplace=True)
    features_all_df.to_csv(out / 'features_non_dl.csv', index=False)

    X = np.stack(all_sequences, axis=0)
    y = np.array(all_labels)
    np.savez_compressed(out / 'dl_data.npz', X=X, y=y)

    print(f"Saved {len(all_sequences)} segments: features for non-DL (CSV) and sequences for DL (NPZ) to {out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocessing Pipeline für Einzeldateien (Speicherschonend)')
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