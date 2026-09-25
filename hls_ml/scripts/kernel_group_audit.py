import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

from config import RAW_DATASET_PATH as DATA_PATH, RESULTS_DIR


def code_hash(source_code):
    """
    Create a stable hash of the stored source code.
    This helps detect whether multiple records share the same
    underlying source program.
    """
    if not source_code:
        return None

    parts = []

    for item in source_code:
        filename = str(item.get("file_name", ""))
        content = str(item.get("file_content", ""))
        parts.append(filename)
        parts.append(content)

    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    print("=" * 75)
    print("ForgeHLS Kernel / Group Audit")
    print("=" * 75)

    print("\nLoading dataset...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Group by source + algorithm
    # ---------------------------------------------------------
    groups = defaultdict(list)

    for row in data:
        key = (
            row.get("source_name"),
            row.get("algo_name"),
        )
        groups[key].append(row)

    print(f"Total records: {len(data):,}")
    print(f"Groups: {len(groups):,}")

    # ---------------------------------------------------------
    # Analyze each group
    # ---------------------------------------------------------
    output = []

    for (source, algo), rows in groups.items():

        design_ids = {
            row.get("design_id")
            for row in rows
        }

        parts = {
            row.get("Part")
            for row in rows
        }

        clock_periods = {
            row.get("TargetClockPeriod")
            for row in rows
        }

        code_hashes = {
            code_hash(row.get("source_code"))
            for row in rows
        }

        top_functions = {
            row.get("top_function_name")
            for row in rows
        }

        strategies = Counter(
            row.get("latency-resource-strategy")
            for row in rows
        )

        output.append({
            "source_name": source,
            "algo_name": algo,
            "rows": len(rows),
            "unique_design_ids": len(design_ids),
            "unique_parts": len(parts),
            "unique_clock_periods": len(clock_periods),
            "unique_source_code_hashes": len(code_hashes),
            "unique_top_functions": len(top_functions),
            "strategy_count": len(strategies),
        })

    # ---------------------------------------------------------
    # Sort by number of records
    # ---------------------------------------------------------
    output.sort(
        key=lambda x: x["rows"],
        reverse=True
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print("\nTop 30 groups")
    print("-" * 75)

    for item in output[:30]:
        print(
            f"{item['source_name']:35s} | "
            f"{item['algo_name']:45s} | "
            f"rows={item['rows']:5d} | "
            f"design_ids={item['unique_design_ids']:4d} | "
            f"parts={item['unique_parts']:2d} | "
            f"clocks={item['unique_clock_periods']:2d} | "
            f"codes={item['unique_source_code_hashes']:4d} | "
            f"strategies={item['strategy_count']}"
        )

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------
    print("\nAggregate group statistics")
    print("-" * 75)

    groups_with_multiple_codes = sum(
        x["unique_source_code_hashes"] > 1
        for x in output
    )

    groups_with_multiple_design_ids = sum(
        x["unique_design_ids"] > 1
        for x in output
    )


    print(
        "Groups with >1 source-code variant:",
        groups_with_multiple_codes
    )

    print(
        "Groups with >1 design_id:",
        groups_with_multiple_design_ids
    )

    # ---------------------------------------------------------
    # Save CSV-like text report
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / "kernel_group_audit.csv"

    with report_path.open("w", encoding="utf-8") as f:

        header = [
            "source_name",
            "algo_name",
            "rows",
            "unique_design_ids",
            "unique_parts",
            "unique_clock_periods",
            "unique_source_code_hashes",
            "unique_top_functions",
            "strategy_count",
        ]

        f.write(",".join(header) + "\n")

        for item in output:
            f.write(
                ",".join(
                    str(item[h])
                    for h in header
                ) + "\n"
            )

    print("\nSaved:")
    print(report_path)


if __name__ == "__main__":
    main()
