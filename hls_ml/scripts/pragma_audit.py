"""
pragma_audit.py

Comprehensive audit of all HLS pragmas across the 184,717 ForgeHLS design records.
Discovers:
- Unique pragma directives (e.g. PIPELINE, UNROLL, ARRAY_PARTITION, etc.)
- Associated parameters, factors, and attributes (e.g. complete, cyclic, block, II, factor)
- Frequency per design and dataset-wide distributions
- Verification against the stored 'pragma_number' metadata

Outputs saved to hls_ml/results/
"""

import json
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path
import time

from config import RAW_DATASET_PATH as DATA_PATH, RESULTS_DIR


# Regular expression to extract #pragma HLS directives and clauses
PRAGMA_HLS_REGEX = re.compile(
    r"#\s*pragma\s+HLS\s+([a-zA-Z0-9_]+)(.*?)(?:$|//|/\*)",
    re.IGNORECASE
)


def extract_pragmas_from_source(source_code_list):
    """
    Given a list of file dictionaries with 'file_content',
    returns a list of (raw_line, directive_upper, remainder_clean).
    """
    extracted = []
    if not source_code_list:
        return extracted

    for item in source_code_list:
        content = item.get("file_content", "")
        if not content or "#pragma" not in content.lower():
            continue

        for line in content.splitlines():
            line_str = line.strip()
            match = PRAGMA_HLS_REGEX.search(line_str)
            if match:
                directive = match.group(1).upper()
                remainder = match.group(2).strip()
                extracted.append((line_str, directive, remainder))

    return extracted


def main():
    print("=" * 70)
    print("ForgeHLS Pragma Audit (Milestone 1 - Step 1.1)")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print("\n[1] Loading dataset JSON...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data)
    load_time = time.time() - start_time
    print(f"Loaded {total_records:,} records in {load_time:.2f}s")

    print("\n[2] Auditing pragmas across all designs...")
    scan_start = time.time()

    directive_counts = Counter()
    directive_presence_in_designs = Counter()  # in how many designs does it appear at least once
    full_pragma_patterns = Counter()           # canonical pragma line templates
    pragmas_per_design = Counter()              # distribution of pragma count per design
    stored_pragma_number_match = 0
    stored_vs_extracted_diffs = 0

    # Attributes tracking
    pipeline_types = Counter()                  # e.g., default, II=1, off, rewind
    unroll_factors = Counter()                  # e.g., complete, factor=2, 4, 8, etc.
    array_partition_types = Counter()           # complete, cyclic, block, etc.

    for idx, row in enumerate(data):
        source_code = row.get("source_code", [])
        stored_pragma_num = row.get("pragma_number", 0)

        pragmas = extract_pragmas_from_source(source_code)
        num_pragmas = len(pragmas)
        pragmas_per_design[num_pragmas] += 1

        if num_pragmas == stored_pragma_num:
            stored_pragma_number_match += 1
        else:
            stored_vs_extracted_diffs += 1

        seen_in_this_design = set()

        for raw_line, directive, remainder in pragmas:
            directive_counts[directive] += 1
            seen_in_this_design.add(directive)

            # Clean full pattern
            cleaned_pattern = f"#pragma HLS {directive} {remainder}".strip()
            full_pragma_patterns[cleaned_pattern] += 1

            # Specific directive parameter inspections
            if directive == "PIPELINE":
                rem_lower = remainder.lower()
                if "off" in rem_lower:
                    pipeline_types["off"] += 1
                elif "ii=" in rem_lower:
                    match_ii = re.search(r"ii\s*=\s*(\d+)", rem_lower)
                    if match_ii:
                        pipeline_types[f"II={match_ii.group(1)}"] += 1
                    else:
                        pipeline_types["II_specified"] += 1
                elif "rewind" in rem_lower:
                    pipeline_types["rewind"] += 1
                else:
                    pipeline_types["default/enable"] += 1

            elif directive == "UNROLL":
                rem_lower = remainder.lower()
                if "factor=" in rem_lower:
                    match_f = re.search(r"factor\s*=\s*(\d+)", rem_lower)
                    if match_f:
                        unroll_factors[f"factor_{match_f.group(1)}"] += 1
                    else:
                        unroll_factors["factor_custom"] += 1
                elif "skip_exit_check" in rem_lower:
                    unroll_factors["skip_exit_check"] += 1
                else:
                    unroll_factors["complete"] += 1

            elif directive == "ARRAY_PARTITION":
                rem_lower = remainder.lower()
                for ptype in ["complete", "cyclic", "block"]:
                    if ptype in rem_lower:
                        array_partition_types[ptype] += 1
                        break
                else:
                    array_partition_types["unspecified_or_other"] += 1

        for d in seen_in_this_design:
            directive_presence_in_designs[d] += 1

        if (idx + 1) % 50000 == 0:
            elapsed = time.time() - scan_start
            print(f"  Processed {idx + 1:,}/{total_records:,} designs ({elapsed:.1f}s)...")

    total_scan_time = time.time() - scan_start
    print(f"Scan complete in {total_scan_time:.2f}s")

    # -------------------------------------------------------------------------
    # [3] Save CSV Reports
    # -------------------------------------------------------------------------
    csv_file = RESULTS_DIR / "pragma_frequency.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["directive", "total_occurrences", "designs_with_directive", "design_percentage"])
        for directive, total_occ in directive_counts.most_common():
            designs_count = directive_presence_in_designs[directive]
            pct = (designs_count / total_records) * 100.0
            writer.writerow([directive, total_occ, designs_count, f"{pct:.2f}%"])

    print(f"\n[3] Saved pragma frequency table to: {csv_file}")

    # -------------------------------------------------------------------------
    # [4] Write Text Summary Report
    # -------------------------------------------------------------------------
    summary_file = RESULTS_DIR / "pragma_audit_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("ForgeHLS Pragma Audit Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total design records audited: {total_records:,}\n")
        f.write(f"Designs where extracted pragma count == stored 'pragma_number': {stored_pragma_number_match:,} ({(stored_pragma_number_match/total_records)*100:.2f}%)\n")
        f.write(f"Designs with mismatch (e.g. non-HLS pragmas or multi-file headers): {stored_vs_extracted_diffs:,}\n\n")

        f.write("1. Unique Pragma Directives Discovered (Ranked by Occurrences):\n")
        f.write("-" * 70 + "\n")
        for directive, count in directive_counts.most_common():
            pct_designs = (directive_presence_in_designs[directive] / total_records) * 100
            f.write(f"  - {directive:<20}: {count:>8,} total times across {directive_presence_in_designs[directive]:>7,} designs ({pct_designs:>6.2f}% of designs)\n")

        f.write("\n2. Pragma Count per Design Distribution:\n")
        f.write("-" * 70 + "\n")
        for count, designs_with_count in sorted(pragmas_per_design.items()):
            pct = (designs_with_count / total_records) * 100
            f.write(f"  - {count:>3} pragmas: {designs_with_count:>7,} designs ({pct:>6.2f}%)\n")

        f.write("\n3. Pipeline Pragma Breakdown:\n")
        f.write("-" * 70 + "\n")
        for ptype, count in pipeline_types.most_common():
            f.write(f"  - {ptype:<20}: {count:>8,}\n")

        f.write("\n4. Unroll Pragma Breakdown:\n")
        f.write("-" * 70 + "\n")
        for factor, count in unroll_factors.most_common():
            f.write(f"  - {factor:<20}: {count:>8,}\n")

        f.write("\n5. Array Partition Breakdown:\n")
        f.write("-" * 70 + "\n")
        for part_type, count in array_partition_types.most_common():
            f.write(f"  - {part_type:<20}: {count:>8,}\n")

        f.write("\n6. Top 20 Most Frequent Full Pragma Patterns:\n")
        f.write("-" * 70 + "\n")
        for pattern, count in full_pragma_patterns.most_common(20):
            f.write(f"  - [{count:>7,}x] {pattern}\n")

    print(f"[4] Saved detailed text summary to: {summary_file}")

    # Print short summary to console
    print("\n" + "=" * 70)
    print("AUDIT FINDINGS HIGHLIGHTS:")
    print("=" * 70)
    for directive, count in directive_counts.most_common(10):
        print(f"  {directive:<18} : {count:>7,} occurrences ({directive_presence_in_designs[directive]:>6,} designs)")
    print("=" * 70)


if __name__ == "__main__":
    main()
