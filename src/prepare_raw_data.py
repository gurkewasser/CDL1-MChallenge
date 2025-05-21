
import zipfile
import pandas as pd
from tqdm import tqdm
import rootutils
import hashlib
import shutil

# Setup project root
root = rootutils.setup_root(__file__, pythonpath=True, cwd=True)

INPUT_DIR = root / "data" / "raw"
UNPACK_DIR = root / "data" / "unpacked"
OUTPUT_DIR = root / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "raw_data.parquet"

# Define relevant sensors and columns
SENSOR_FILES = {
    "Accelerometer.csv": ["x", "y", "z"],
    "Magnetometer.csv": ["x", "y", "z"],
    "Gravity.csv": ["x", "y", "z"],
    "Gyroscope.csv": ["x", "y", "z"],
    "Orientation.csv": ["pitch", "roll", "yaw", "qw", "qx", "qy", "qz"]
}

METADATA_FILE = "Metadata.csv"

def extract_nested_zip_files():
    """
    Unpacks the massive .zip file in INPUT_DIR, then unzips each individual .zip inside it
    into UNPACK_DIR, one folder per recording.
    """
    UNPACK_DIR.mkdir(parents=True, exist_ok=True)
    # Find the massive zip file (assume only one in INPUT_DIR)
    massive_zips = list(INPUT_DIR.glob("*.zip"))
    if not massive_zips:
        print("No .zip file found in", INPUT_DIR)
        return
    massive_zip_path = massive_zips[0]
    # Unpack the massive zip to a temp folder
    temp_extract_dir = INPUT_DIR / "temp_massive_unzip"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(massive_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    # Now, for each .zip file in the temp folder, extract to UNPACK_DIR/<recording_name>
    for rec_zip in temp_extract_dir.glob("*.zip"):
        target_folder = UNPACK_DIR / rec_zip.stem
        if not target_folder.exists():
            with zipfile.ZipFile(rec_zip, 'r') as zip_ref:
                zip_ref.extractall(target_folder)
    # Optionally, clean up temp folder
    # Remove all files and directories inside temp_extract_dir
    for f in temp_extract_dir.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)
    temp_extract_dir.rmdir()

def get_label_from_filename(filename):
    return filename.split("_")[0]

def load_metadata(folder):
    meta_path = folder / METADATA_FILE
    if meta_path.exists():
        df = pd.read_csv(meta_path)
        meta = {
            "device_id": df.get("device id", [None])[0],
            "device_name": df.get("device name", [None])[0],
            "app_version": df.get("appVersion", [None])[0]
        }
        return meta
    return {"device_id": None, "device_name": None, "app_version": None}

def load_sensor_data(folder):
    dfs = []
    time_column = None
    for fname, columns in SENSOR_FILES.items():
        fpath = folder / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if "seconds_elapsed" not in df.columns or "time" not in df.columns:
            continue
        df["time"] = pd.to_datetime(df["time"], unit="ns")
        if time_column is None:
            time_column = df[["seconds_elapsed", "time"]].copy()
        df = df[["seconds_elapsed"] + columns].copy()
        for col in columns:
            df.rename(columns={col: f"{fname.replace('.csv','').lower()}_{col}"}, inplace=True)
        dfs.append(df)
    if not dfs:
        return None
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge_asof(merged.sort_values("seconds_elapsed"),
                               d.sort_values("seconds_elapsed"),
                               on="seconds_elapsed")
    merged = pd.merge_asof(merged.sort_values("seconds_elapsed"),
                           time_column.sort_values("seconds_elapsed"),
                           on="seconds_elapsed")
    return merged

def process_all_zips():
    extract_nested_zip_files()
    raw_rows = []
    # Now, after extraction, each folder in UNPACK_DIR is a session
    # For hash, use the original .zip file for each session (from the temp extraction)
    # But since we deleted temp, skip hash or set to None
    for folder in tqdm(list(UNPACK_DIR.iterdir()), desc="Processing sessions"):
        label = get_label_from_filename(folder.name)
        file_hash = None  # Could be added if needed
        metadata = load_metadata(folder)
        sensor_df = load_sensor_data(folder)
        if sensor_df is None:
            continue
        sensor_df["label"] = label
        sensor_df["device_id"] = metadata["device_id"]
        sensor_df["device_name"] = metadata["device_name"]
        sensor_df["app_version"] = metadata["app_version"]
        sensor_df["file_hash"] = file_hash
        raw_rows.append(sensor_df)
    if raw_rows:
        df_all = pd.concat(raw_rows, ignore_index=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(OUTPUT_PATH, index=False)
        print(f"✅ Saved raw data to {OUTPUT_PATH}")
    else:
        print("⚠️ No valid data found.")

if __name__ == "__main__":
    process_all_zips()
