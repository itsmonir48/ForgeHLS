"""
test_feature_extractor.py
Unit test and benchmark for feature_extractor.py on actual ForgeHLS records.
"""

import json
import time
from pathlib import Path
from feature_extractor import (
    extract_features_from_record,
    extract_targets_from_record,
    FEATURES_A,
    FEATURES_B,
    FEATURES_C,
    TARGET_COLUMNS
)
from config import RAW_DATASET_PATH as DATA_PATH

def main():
    print("=" * 60)
    print("Testing Feature Extractor (Milestone 1 - Step 1.2 Verification)")
    print("=" * 60)

    print(f"Feature Set A count: {len(FEATURES_A)}")
    print(f"Feature Set B count: {len(FEATURES_B)}")
    print(f"Feature Set C count: {len(FEATURES_C)}")
    print(f"Target count:        {len(TARGET_COLUMNS)}")

    print("\n[1] Reading sample records from dataset...")
    # Load first 500 records
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data[:500]
    print(f"Loaded {len(samples)} sample records for testing.")

    print("\n[2] Extracting features and targets...")
    start_time = time.time()
    for i, row in enumerate(samples):
        feats = extract_features_from_record(row)
        targets = extract_targets_from_record(row)

        # Assert all keys are present
        for k in FEATURES_C:
            assert k in feats, f"Missing feature {k} in row {i}"
            assert isinstance(feats[k], (int, float)), f"Feature {k} is not numeric in row {i}"
            assert not (feats[k] != feats[k]), f"Feature {k} is NaN in row {i}"

        for t in TARGET_COLUMNS:
            assert t in targets, f"Missing target {t} in row {i}"
            assert isinstance(targets[t], (int, float)), f"Target {t} is not numeric in row {i}"
            assert not (targets[t] != targets[t]), f"Target {t} is NaN in row {i}"

    elapsed = time.time() - start_time
    rate = len(samples) / elapsed
    print(f"Verification SUCCESS! Processed {len(samples)} rows in {elapsed:.3f}s ({rate:.1f} rows/sec).")

    # Print sample feature extraction from first record
    sample_feat = extract_features_from_record(samples[0])
    sample_tgt = extract_targets_from_record(samples[0])
    print("\nSample row extraction:")
    print("  Targets:", sample_tgt)
    print("  Feature Set A:", {k: sample_feat[k] for k in FEATURES_A})
    print("  Sample Pragma Feats (Set B):", {k: sample_feat[k] for k in ["pragma_pipeline_enable", "pragma_unroll_count", "pragma_max_unroll_factor", "pragma_array_partition_cyclic"]})
    print("  Sample Structural Feats (Set C):", {k: sample_feat[k] for k in ["code_line_count", "code_function_count", "code_total_loop_count", "code_max_nesting_depth"]})
    check_known_values()
    print("\nVerification 100% PASSED!")


def check_known_values():
    """Exact-value checks on a hand-written design, not just 'it runs'."""
    code = "\n".join([
        "void top(int a[64], int b[64]) {",
        "#pragma HLS ARRAY_PARTITION variable=a cyclic factor=4",
        "#pragma HLS ARRAY_PARTITION variable=b complete",
        "  for (int i = 0; i < 64; i++) {",
        "#pragma HLS PIPELINE II=1",
        "#pragma HLS UNROLL factor=8",
        "    b[i] = a[i] * 2;",
        "  }",
        "  for (int j = 0; j < 4; j++) {",
        "#pragma HLS UNROLL",
        "    b[j] += 1;",
        "  }",
        "}",
    ])
    row = {
        "TargetClockPeriod": 10, "code_length": len(code), "pragma_number": 5,
        "source_code": [{"file_content": code}],
    }
    f = extract_features_from_record(row)
    expected = {
        "pragma_pipeline_count": 1, "pragma_pipeline_enable": 1,
        "pragma_unroll_count": 2, "pragma_unroll_factor_8": 1,
        # factor-less UNROLL is a complete unroll: counted in the >=16 bucket
        # and reported with the sentinel, never as factor 0.
        "pragma_unroll_factor_16": 1, "pragma_max_unroll_factor": 1024,
        "pragma_array_partition_count": 2, "pragma_array_partition_cyclic": 1,
        "pragma_array_partition_complete": 1, "pragma_max_partition_factor": 4,
        "code_for_loop_count": 2, "code_function_count": 1,
    }
    for key, value in expected.items():
        assert f[key] == value, f"{key}: expected {value}, got {f[key]}"
    print("  Known-value checks on a hand-written design: OK")


if __name__ == "__main__":
    main()
