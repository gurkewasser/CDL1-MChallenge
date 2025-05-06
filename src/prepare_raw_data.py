
import zipfile
import pandas as pd
from tqdm import tqdm
import rootutils
import hashlib

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

def extract_zip_files():
    UNPACK_DIR.mkdir(parents=True, exist_ok=True)
    for zip_path in INPUT_DIR.glob("*.zip"):
        target_folder = UNPACK_DIR / zip_path.stem
        if not target_folder.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_folder)

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
    extract_zip_files()
    raw_rows = []
    zip_path_lookup = {f.stem: f for f in INPUT_DIR.glob("*.zip")}
    for folder in tqdm(list(UNPACK_DIR.iterdir()), desc="Processing sessions"):
        label = get_label_from_filename(folder.name)
        # Find the corresponding zip file and compute its hash
        zip_path = zip_path_lookup.get(folder.name)
        if zip_path and zip_path.exists():
            with open(zip_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        else:
            file_hash = None
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
