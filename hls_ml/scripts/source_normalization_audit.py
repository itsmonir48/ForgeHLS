import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path

from config import RAW_DATASET_PATH as DATA_PATH, RESULTS_DIR


def get_source_text(source_code):
    if not source_code:
        return ""

    parts = []

    for item in source_code:
        filename = str(item.get("file_name", ""))
        content = str(item.get("file_content", ""))

        parts.append(f"FILE:{filename}")
        parts.append(content)

    return "\n".join(parts)


def remove_hls_pragmas(text):
    """
    Remove HLS pragma lines.

    This is intentionally conservative:
    only lines containing '#pragma HLS' are removed.
    """
    lines = text.splitlines()

    cleaned = []

    for line in lines:
        if re.search(r"#\s*pragma\s+HLS\b", line, re.IGNORECASE):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def normalize_whitespace(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def hash_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():
    print("=" * 75)
    print("ForgeHLS Source Normalization Audit")
    print("=" * 75)

    print("\nLoading dataset...")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = defaultdict(list)

    for row in data:
        key = (
            row.get("source_name"),
            row.get("algo_name"),
        )
        groups[key].append(row)

    results = []

    for (source, algo), rows in groups.items():

        raw_hashes = set()
        normalized_hashes = set()

        pragma_counts = []

        for row in rows:
            source_text = get_source_text(row.get("source_code"))

            raw_hashes.add(
                hash_text(source_text)
            )

            normalized = remove_hls_pragmas(source_text)
            normalized = normalize_whitespace(normalized)

            normalized_hashes.add(
                hash_text(normalized)
            )

            pragma_counts.append(
                row.get("pragma_number")
            )

        results.append({
            "source": source,
            "algo": algo,
            "rows": len(rows),
            "raw_code_variants": len(raw_hashes),
            "code_variants_after_pragma_removal": len(normalized_hashes),
            "min_pragma": min(pragma_counts),
            "max_pragma": max(pragma_counts),
        })

    results.sort(
        key=lambda x: x["rows"],
        reverse=True
    )

    print("\nTop 30 groups")
    print("-" * 75)

    for r in results[:30]:
        print(
            f"{r['source']:35s} | "
            f"{r['algo']:45s} | "
            f"rows={r['rows']:5d} | "
            f"raw={r['raw_code_variants']:5d} | "
            f"after_pragma_removal={r['code_variants_after_pragma_removal']:5d} | "
            f"pragma={r['min_pragma']}-{r['max_pragma']}"
        )

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------

    same_base_after_pragma_removal = sum(
        r["code_variants_after_pragma_removal"] == 1
        for r in results
        if r["rows"] > 1
    )

    multiple_base_after_pragma_removal = sum(
        r["code_variants_after_pragma_removal"] > 1
        for r in results
        if r["rows"] > 1
    )

    print("\nAggregate")
    print("-" * 75)

    print(
        "Multi-row groups with ONE base source after pragma removal:",
        same_base_after_pragma_removal
    )

    print(
        "Multi-row groups with MULTIPLE base sources after pragma removal:",
        multiple_base_after_pragma_removal
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = (
        RESULTS_DIR /
        "source_normalization_audit.csv"
    )

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        headers = [
            "source",
            "algo",
            "rows",
            "raw_code_variants",
            "code_variants_after_pragma_removal",
            "min_pragma",
            "max_pragma",
        ]

        f.write(",".join(headers) + "\n")

        for r in results:
            f.write(
                ",".join(
                    str(r[h])
                    for h in headers
                ) + "\n"
            )

    print("\nSaved:")
    print(report_path)


if __name__ == "__main__":
    main()