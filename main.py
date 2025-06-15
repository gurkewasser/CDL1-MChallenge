from pathlib import Path
import pandas as pd
import numpy as np
import sys
import zipfile
import tempfile
from tqdm import tqdm
import os
import shutil

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

print(f"Project root: {project_root}")

TEST_SIZE = 0.1
SEED = 2277

from src.methods import *

def run_preprocessing_pipeline(
    zip_dir: str,
    output_dir: str,
    target_hz: int = 50,
    apply_augment: bool = False,
    noise_level: float = 0.2,
    orientation_aug_methods: list = None,
    val_set: bool = False
):
    zip_dir = Path(zip_dir)
    output_dir = Path(output_dir)
    unzipped_raw_path = output_dir / "unzipped_raw"
    unzipped_raw_created = False

    # Entpacke die data.zip, wenn sie im Verzeichnis liegt
    single_zip = list(zip_dir.glob("*.zip"))
    if len(single_zip) == 1 and single_zip[0].name == "data.zip":
        unzipped_raw_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(single_zip[0], "r") as zip_ref:
            zip_ref.extractall(unzipped_raw_path)
        data_folder = unzipped_raw_path / "data"
        zip_dir = data_folder if data_folder.exists() else unzipped_raw_path
        unzipped_raw_created = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unzip top-level archive if necessary
    if zip_dir.suffix == ".zip":
        print(f"Unzipping {zip_dir} to {output_dir}")
        unzipped_raw_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(unzipped_raw_path)
        # If a 'data' subfolder exists after extraction, use it
        data_folder = unzipped_raw_path / "data"
        if data_folder.exists() and data_folder.is_dir():
            zip_dir = data_folder
        else:
            zip_dir = unzipped_raw_path
        unzipped_raw_created = True
        

    # Scan for valid zip files inside the extracted directory (excluding the outer "data.zip" and macOS metadata)
    zip_files = [
        f for f in zip_dir.rglob("*.zip")
        if get_label_from_filename(f.name) != "data"
        and "__MACOSX" not in f.parts
        and not f.name.startswith("._")
        and zipfile.is_zipfile(str(f))
    ]

    if val_set:
        train_zips, val_zips,  test_zips = create_stratified_split(zip_files, test_size=TEST_SIZE, seed=SEED, create_val=val_set)
    else:
        train_zips, test_zips = create_stratified_split(zip_files, test_size=TEST_SIZE, seed=SEED)

    def process_and_extract(zip_files, augment=False, orientation_method=None):
        all_segments = []
        for zip_file in tqdm(zip_files, desc=f"Processing ({'aug' if augment else 'base'}){f' + {orientation_method}' if orientation_method else ''}"):
            df = load_and_merge_sensor_data(zip_file)

            metadata_path = extract_metadata(zip_file)
            df = standardize_coordinates(df, metadata_path)
            df = trim_data(df)
            df = resample_data(df, target_hz=target_hz)

            if orientation_method:
                df = augment_orientation_pose(df, method=orientation_method)

            if augment:
                df = apply_augmentation(df, noise_level=noise_level)

            df = apply_moving_average(df)
            df["label"] = get_label_from_filename(zip_file.name)

            segments = segment_dataframe(df)
            all_segments.extend(segments)

        return all_segments

    def export(segments, suffix):
        (output_dir / "NDL").mkdir(parents=True, exist_ok=True)
        (output_dir / "DL").mkdir(parents=True, exist_ok=True)

        # Feature-Namen aus dem ersten Segment holen
        feature_names = segments[0]["feature_names"] if segments else None

        # Klassische ML-Features mit aussagekräftigen Namen
        df_feat = extract_features_from_segments(segments, feature_names=feature_names)
        df_feat.to_csv(output_dir / "NDL" / f"features_{suffix}.csv", index=False)

        # Deep Learning Daten
        X = np.array([s["data"] for s in segments])
        y = np.array([s["label"] for s in segments])
        np.savez_compressed(output_dir / "DL" / f"dl_{suffix}.npz", X=X, y=y)

    try:
        # 1. Normal
        seg_train = process_and_extract(train_zips)
        export(seg_train, "train")
        seg_test = process_and_extract(test_zips)
        export(seg_test, "test")
        seg_val = process_and_extract(val_zips)
        export(seg_val, "val")

        # 2. Mit Rauschaugmentation
        seg_train_aug = process_and_extract(train_zips, augment=True)
        export(seg_train_aug, "train_aug")

        # 3. Mit verschiedenen Orientation-Augs
        for method in ["flip_roll", "invert_pitch", "rotate_yaw_180"]:
            seg_aug = process_and_extract(train_zips, orientation_method=method)
            export(seg_aug, f"train_{method}")
    finally:
        # Clean up unzipped_raw if it was created
        if unzipped_raw_created and unzipped_raw_path.exists():
            print(f"Cleaning up...")
            try:
                shutil.rmtree(unzipped_raw_path)
            except Exception as e:
                print(f"⚠️ Fehler beim Löschen von {unzipped_raw_path}: {e}")

def extract_metadata(zip_file):
    try:
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if "Metadata.csv" in zip_ref.namelist():
                zip_ref.extract("Metadata.csv", tmpdir)
                metadata_path = Path(tmpdir) / "Metadata.csv"
                if metadata_path.exists():
                    return metadata_path
                else:
                    print(f"⚠️ Metadata.csv nicht gefunden nach dem Extrahieren aus {zip_file.name}")
                    return None
            else:
                print(f"⚠️ Keine Metadata.csv in {zip_file.name}")
                return None
    except zipfile.BadZipFile:
        print(f"⚠️ Not a zip file or corrupted: {zip_file}")
        return None
    except Exception as e:
        print(f"⚠️ Fehler beim Extrahieren der Metadata.csv aus {zip_file.name}: {e}")
        return None


if __name__ == "__main__":
    run_preprocessing_pipeline(
        zip_dir="data/raw",
        output_dir="data/processed",
        target_hz=50,
        apply_augment=True,
        noise_level=2.0,
        orientation_aug_methods=["flip_roll", "invert_pitch", "rotate_yaw_180"],
        val_set=True
    )