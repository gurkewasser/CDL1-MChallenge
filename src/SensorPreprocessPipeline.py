import os
import zipfile
import rootutils
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob
from scipy.signal import savgol_filter
from datetime import timedelta


root = rootutils.setup_root(__file__, pythonpath=True, cwd=True)

INPUT_DIR      = root / "data" / "raw"         # where your .zip files live
OUTPUT_DIR     = root / "data" / "processed"   # where processed outputs go
CROP_SECONDS   = 5                             # crop first/last seconds
RESAMPLE_RATE  = "50ms"                         # e.g. 50 ms = 20 Hz
WINDOW_SIZE    = 5.0                           # seconds per window
WINDOW_STEP    = 2.5                           # seconds between window starts
TARGET_CSVS    = ["Accelerometer.csv",
                  "Gyroscope.csv",
                  "Magnetometer.csv",
                  "Gravity.csv",
                  "Orientation.csv"]

# derived dirs
TS_DIR     = OUTPUT_DIR / "timeseries"
WIN_DIR    = OUTPUT_DIR / "windows"


def extract_and_filter(zip_path, target_files):
    """
    Open zip and read only the CSV files in target_files,
    returning a dict: { sensor_name_without_ext: DataFrame, ... }.
    """
    dfs = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for fname in target_files:
            try:
                with zf.open(fname) as f:
                    df = pd.read_csv(f)
                    key = Path(fname).stem
                    dfs[key] = df
            except KeyError:
                # some zips may lack optional files
                continue
    return dfs


def preprocess_signal(df, crop=CROP_SECONDS, resample_rate=RESAMPLE_RATE):
    """
    1) detect time column (case-insensitive 'time')
    2) convert to datetime (ms vs. ns heuristics)
    3) set as index
    4) crop first/last N seconds
    5) resample + interpolate
    """
    # detect time column
    time_cols = [c for c in df.columns if c.lower() == "time"]
    if not time_cols:
        raise KeyError("No 'Time' column found in DataFrame")
    tcol = time_cols[0]

    orig = df[tcol]
    # heuristics for unit
    if pd.api.types.is_integer_dtype(orig) or pd.api.types.is_float_dtype(orig):
        unit = "ns" if orig.max() > 1e12 else "ms"
    else:
        unit = None

    # convert
    if unit:
        df[tcol] = pd.to_datetime(orig, unit=unit)
    else:
        df[tcol] = pd.to_datetime(orig)

    df = df.set_index(tcol)

    # crop
    start = df.index[0] + timedelta(seconds=crop)
    end   = df.index[-1] - timedelta(seconds=crop)
    df = df.loc[start : end]

    # resample & interpolate
    df = df.resample(resample_rate).mean().interpolate()
    return df


def smooth_df(df, window_length=11, polyorder=2):
    """
    Savitzky–Golay filter per column.
    Ensures window_length odd and <= n_samples.
    """
    smooth = df.copy()
    n = len(df)
    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 3)

    for col in df.columns:
        smooth[col] = savgol_filter(df[col], wl, polyorder)
    return smooth


def sliding_windows(df, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """
    Split df.values into overlapping windows of fixed length.
    Returns list of np.ndarray, each shape (n_samples_per_window, n_features).
    """
    freq_s = pd.to_timedelta(RESAMPLE_RATE).total_seconds()
    n_samples = int(window_size / freq_s)
    step_samples = int(step / freq_s)

    arr = df.values
    n    = arr.shape[0]
    windows = []
    for start in range(0, n - n_samples + 1, step_samples):
        end = start + n_samples
        windows.append(arr[start:end])
    return windows


def extract_features(window, col_names):
    """
    From one window (2D array), compute basic stats per column.
    Returns dict of feature_name: value.
    """
    feats = {}
    for i, name in enumerate(col_names):
        series = window[:, i]
        feats[f"{name}_mean"]   = np.mean(series)
        feats[f"{name}_std"]    = np.std(series)
        feats[f"{name}_min"]    = np.min(series)
        feats[f"{name}_max"]    = np.max(series)
        feats[f"{name}_median"] = np.median(series)
    return feats



if __name__ == "__main__":
    # make sure output dirs exist
    for d in (OUTPUT_DIR, TS_DIR, WIN_DIR):
        d.mkdir(parents=True, exist_ok=True)

    zips = glob(str(INPUT_DIR / "*.zip"))
    feature_rows = []

    # optional mapping to German labels
    activity_map = {
        "walking":  "Laufen",
        "jogging":  "Joggen",
        "sitting":  "Sitzen",
        "standing": "Stehen"
    }

    for zip_path in zips:
        raw = Path(zip_path).stem.split("_")[0]             # e.g. "jogging"
        label = activity_map.get(raw.lower(), raw)

        # 1) extract only needed CSVs
        dfs = extract_and_filter(zip_path, TARGET_CSVS)
        if not dfs:
            continue

        # 2) preprocess & smooth each sensor
        prepped = {}
        for sensor, df in dfs.items():
            df_p = preprocess_signal(df)
            df_s = smooth_df(df_p)
            prepped[sensor] = df_s

        # 3) merge on timestamp
        merged = pd.concat(prepped.values(), axis=1, join="inner")

        # save full processed time series per recording
        merged.to_csv(TS_DIR / f"{raw}_processed.csv", index=True)

        # 4) segment into windows
        windows = sliding_windows(merged)

        # save windows as a stacked .npy array for DL models
        np.save(WIN_DIR / f"{raw}_windows.npy", np.stack(windows))

        # 5) extract features for each window (for RF, SVM, KNN, MLP)
        for w in windows:
            feats = extract_features(w, merged.columns)
            feats["label"] = label
            feature_rows.append(feats)

    # 6) save combined feature matrix
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(OUTPUT_DIR / "features.csv", index=False)

    print("Processing complete.")
    print(f"- full time series: {len(zips)} files → {TS_DIR}")
    print(f"- windows (.npy):        {len(zips)} files → {WIN_DIR}")
    print(f"- feature matrix:        {features_df.shape} → features.csv")