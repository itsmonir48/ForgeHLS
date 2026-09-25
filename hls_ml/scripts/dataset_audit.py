import json
from collections import Counter, defaultdict
from pathlib import Path

from config import RAW_DATASET_PATH as DATA_PATH, RESULTS_DIR


def main():
    print("=" * 70)
    print("ForgeHLS Dataset Audit")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading dataset...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records: {len(data):,}")

    # ---------------------------------------------------------
    # Basic counts
    # ---------------------------------------------------------
    sources = Counter(row.get("source_name") for row in data)
    algorithms = Counter(row.get("algo_name") for row in data)
    design_ids = Counter(row.get("design_id") for row in data)
    strategies = Counter(
        row.get("latency-resource-strategy") for row in data
    )

    source_algo_groups = Counter(
        (row.get("source_name"), row.get("algo_name"))
        for row in data
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n[2] Dataset summary")
    print("-" * 50)
    print(f"Unique sources           : {len(sources):,}")
    print(f"Unique algorithms        : {len(algorithms):,}")
    print(f"Unique design IDs        : {len(design_ids):,}")
    print(f"Unique source+algorithm  : {len(source_algo_groups):,}")

    # ---------------------------------------------------------
    # Pareto / kernel fields
    # ---------------------------------------------------------
    print("\n[3] Special fields")
    print("-" * 50)

    is_pareto = Counter(row.get("is_pareto") for row in data)
    is_kernel = Counter(row.get("is_kernel") for row in data)

    print(f"is_pareto: {dict(is_pareto)}")
    print(f"is_kernel: {dict(is_kernel)}")

    # ---------------------------------------------------------
    # Strategies
    # ---------------------------------------------------------
    print("\n[4] Strategy distribution")
    print("-" * 50)

    for strategy, count in strategies.most_common():
        print(f"{str(strategy):35s} : {count:,}")

    # ---------------------------------------------------------
    # Top algorithms
    # ---------------------------------------------------------
    print("\n[5] Top 20 algorithms")
    print("-" * 50)

    for algo, count in algorithms.most_common(20):
        print(f"{str(algo):50s} : {count:,}")

    # ---------------------------------------------------------
    # Top source+algorithm groups
    # ---------------------------------------------------------
    print("\n[6] Top 20 source+algorithm groups")
    print("-" * 50)

    for (source, algo), count in source_algo_groups.most_common(20):
        print(
            f"{str(source):35s} | "
            f"{str(algo):45s} | "
            f"{count:,}"
        )

    # ---------------------------------------------------------
    # Duplicate design IDs
    # ---------------------------------------------------------
    repeated_design_ids = {
        design_id: count
        for design_id, count in design_ids.items()
        if count > 1
    }

    print("\n[7] Design ID repetition")
    print("-" * 50)
    print(f"Design IDs repeated: {len(repeated_design_ids):,}")
    print(
        f"Maximum rows for one design ID: "
        f"{max(design_ids.values()):,}"
    )


    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    fields = [
        "Part",
        "TargetClockPeriod",
        "Best-caseLatency",
        "Worst-caseLatency",
        "BRAM_18K",
        "LUT",
        "DSP",
        "FF",
        "design_id",
        "algo_name",
        "source_name",
        "source_code",
        "code_length",
        "pragma_number",
        "is_pareto",
        "is_kernel",
        "top_function_name",
        "latency-resource-strategy",
    ]

    print("\n[8] Missing values")
    print("-" * 50)

    missing = {}

    for field in fields:
        count = sum(row.get(field) is None for row in data)
        missing[field] = count
        print(f"{field:30s}: {count:,}")

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------
    summary_file = RESULTS_DIR / "dataset_audit_summary.txt"

    with summary_file.open("w", encoding="utf-8") as f:
        f.write("ForgeHLS Dataset Audit\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total records: {len(data):,}\n")
        f.write(f"Unique sources: {len(sources):,}\n")
        f.write(f"Unique algorithms: {len(algorithms):,}\n")
        f.write(f"Unique design IDs: {len(design_ids):,}\n")
        f.write(
            f"Unique source+algorithm groups: "
            f"{len(source_algo_groups):,}\n"
        )

        f.write("\nStrategy distribution:\n")
        for strategy, count in strategies.most_common():
            f.write(f"{strategy}: {count:,}\n")

        f.write("\nis_pareto:\n")
        for key, count in is_pareto.items():
            f.write(f"{key}: {count:,}\n")

        f.write("\nis_kernel:\n")
        for key, count in is_kernel.items():
            f.write(f"{key}: {count:,}\n")

        f.write("\nMissing values:\n")
        for field, count in missing.items():
            f.write(f"{field}: {count:,}\n")

    print("\n[9] Saved:")
    print(summary_file)

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()
