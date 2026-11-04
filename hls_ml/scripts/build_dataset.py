"""
build_dataset.py

Constructs cached tabular datasets from raw ForgeHLS records using feature_extractor.py.
Outputs saved to hls_ml/data/:
- dataset_features.parquet & dataset_targets.parquet & dataset_metadata.parquet
- dataset_arrays.npz (Fast zero-overhead numpy arrays for ML models)
- dataset_schema.json (field mapping for Set A, Set B, Set C, targets)
"""

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from feature_extractor import (
    extract_features_from_record,
    extract_targets_from_record,
    FEATURES_A,
    FEATURES_B,
    FEATURES_C,
    TARGET_COLUMNS
)

from config import RAW_DATASET_PATH as DATA_PATH, DATA_DIR as OUTPUT_DIR


def main():
    print("=" * 70)
    print("Building Tabular Dataset (Milestone 2 - Step 2.1)")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading raw dataset JSON...")
    t0 = time.time()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    total = len(raw_data)
    print(f"Loaded {total:,} records in {time.time() - t0:.2f}s")

    print("\n[2] Extracting features, targets, and metadata...")
    features_list = []
    targets_list = []
    metadata_list = []

    t_start = time.time()
    for i, row in enumerate(raw_data):
        for field in ("source_name", "algo_name", "source_code"):
            if field not in row or row[field] in (None, "", []):
                raise ValueError(f"Record {i} is missing required field: {field}")
        feats = extract_features_from_record(row)
        targets = extract_targets_from_record(row)

        source_name = str(row["source_name"])
        algo_name = str(row["algo_name"])
        group_id = f"{source_name}::{algo_name}"
        is_pareto = bool(row.get("is_pareto", False))

        features_list.append(feats)
        targets_list.append(targets)
        metadata_list.append({
            "group_id": group_id,
            "source_name": source_name,
            "algo_name": algo_name,
            "is_pareto": is_pareto
        })

        if (i + 1) % 50000 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1:,}/{total:,} designs ({elapsed:.1f}s, {rate:.0f} rows/s)...")

    # -------------------------------------------------------------
    # [3] Convert to DataFrames and Validate
    # -------------------------------------------------------------
    print("\n[3] Converting to DataFrames & validating integrity...")
    df_features = pd.DataFrame(features_list)[FEATURES_C]
    df_targets = pd.DataFrame(targets_list)[TARGET_COLUMNS]
    df_metadata = pd.DataFrame(metadata_list)

    # Integrity Checks
    assert len(df_features) == total, f"Features length mismatch: {len(df_features)} != {total}"
    assert len(df_targets) == total, f"Targets length mismatch: {len(df_targets)} != {total}"
    assert len(df_metadata) == total, f"Metadata length mismatch: {len(df_metadata)} != {total}"

    null_feats = df_features.isnull().sum().sum()
    null_targets = df_targets.isnull().sum().sum()
    assert null_feats == 0, f"Found {null_feats} NaN values in features!"
    assert null_targets == 0, f"Found {null_targets} NaN values in targets!"

    unique_groups = df_metadata["group_id"].nunique()
    print(f"  Integrity confirmed: {total:,} rows, 0 NaNs across all 42 features & 5 targets.")
    print(f"  Identified {unique_groups} distinct logical kernel groups.")

    # -------------------------------------------------------------
    # [4] Save to Disk (Direct PyArrow & Compressed NPZ)
    # -------------------------------------------------------------
    print("\n[4] Saving processed datasets to disk...")
    feat_path = OUTPUT_DIR / "dataset_features.parquet"
    tgt_path = OUTPUT_DIR / "dataset_targets.parquet"
    meta_path = OUTPUT_DIR / "dataset_metadata.parquet"
    npz_path = OUTPUT_DIR / "dataset_arrays.npz"

    # Optimize datatypes
    df_features = df_features.astype(np.float32)
    df_targets = df_targets.astype(np.float32)

    # Direct PyArrow write (avoids pandas 3.14 patch issue)
    pq.write_table(pa.Table.from_pandas(df_features), feat_path)
    pq.write_table(pa.Table.from_pandas(df_targets), tgt_path)
    pq.write_table(pa.Table.from_pandas(df_metadata), meta_path)

    # Also save as compressed NPZ for zero-overhead NumPy loading
    np.savez_compressed(
        npz_path,
        X_features=df_features.values,
        y_targets=df_targets.values,
        # Use a Unicode array so ordinary np.load(..., allow_pickle=False) works.
        group_ids=df_metadata["group_id"].to_numpy(dtype=str),
        is_pareto=df_metadata["is_pareto"].values.astype(bool),
        feature_names=np.array(FEATURES_C),
        target_names=np.array(TARGET_COLUMNS)
    )

    print(f"  Saved Parquet: {feat_path} ({feat_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved Parquet: {tgt_path} ({tgt_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved Parquet: {meta_path} ({meta_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved NPZ:     {npz_path} ({npz_path.stat().st_size / (1024*1024):.2f} MB)")

    # Save schema mapping JSON
    schema_info = {
        "total_records": total,
        "unique_groups": unique_groups,
        "features_set_a": FEATURES_A,
        "features_set_b": FEATURES_B,
        "features_set_c": FEATURES_C,
        "target_columns": TARGET_COLUMNS
    }
    schema_path = OUTPUT_DIR / "dataset_schema.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)
    print(f"  Saved schema reference: {schema_path}")

    print("\n" + "=" * 70)
    print("Dataset construction successfully finished!")
    print("=" * 70)


if __name__ == "__main__":
    main()
