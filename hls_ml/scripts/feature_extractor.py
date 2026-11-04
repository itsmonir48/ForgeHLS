"""
feature_extractor.py

Deterministic feature extraction module for ForgeHLS C/C++ designs.
Extracts:
- Feature Set A (Basic): TargetClockPeriod, code_length, pragma_number
- Feature Set B (Pragma-Aware): Set A + 22 pragma-specific counts and maximum factors
- Feature Set C (Structural + Pragma): Set B + 14 deterministic code-structure metrics

Leakage prevention:
- Excludes: source_name, algo_name, design_id, latency-resource-strategy, is_kernel
"""

import re
import math
from typing import Dict, Any, List, Tuple

# Pre-compiled Regex patterns for Pragma Extraction
PRAGMA_HLS_REGEX = re.compile(
    r"#\s*pragma\s+HLS\s+([a-zA-Z0-9_]+)(.*?)(?:$|//|/\*)",
    re.IGNORECASE
)
FACTOR_REGEX = re.compile(r"factor\s*=\s*(\d+)", re.IGNORECASE)

# Pre-compiled Regex patterns for Structural Code Metrics
COMMENT_BLOCK_REGEX = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE_REGEX = re.compile(r"//.*$", re.MULTILINE)
FOR_LOOP_REGEX = re.compile(r"\bfor\s*\(", re.IGNORECASE)
WHILE_LOOP_REGEX = re.compile(r"\bwhile\s*\(", re.IGNORECASE)
IF_BRANCH_REGEX = re.compile(r"\bif\s*\(", re.IGNORECASE)
SWITCH_BRANCH_REGEX = re.compile(r"\bswitch\s*\(", re.IGNORECASE)
FUNCTION_DEF_REGEX = re.compile(
    r"(?:void|int|float|double|char|short|long|bool|unsigned|int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t|ap_int|ap_uint|ap_fixed|ap_ufixed)\s+([a-zA-Z0-9_]+)\s*\([^;]*?\)\s*\{",
    re.IGNORECASE
)


# Feature Names Definitions
FEATURES_A: List[str] = [
    "TargetClockPeriod",
    "code_length",
    "pragma_number",
]

FEATURES_B_ONLY: List[str] = [
    "pragma_pipeline_count",
    "pragma_pipeline_off",
    "pragma_pipeline_enable",
    "pragma_unroll_count",
    "pragma_unroll_factor_1",
    "pragma_unroll_factor_2",
    "pragma_unroll_factor_4",
    "pragma_unroll_factor_8",
    "pragma_unroll_factor_16",
    "pragma_max_unroll_factor",
    "pragma_array_partition_count",
    "pragma_array_partition_cyclic",
    "pragma_array_partition_complete",
    "pragma_array_partition_block",
    "pragma_max_partition_factor",
    "pragma_loop_tripcount_count",
    "pragma_inline_count",
    "pragma_interface_count",
    "pragma_dataflow_count",
    "pragma_stream_count",
    "pragma_bind_storage_count",
    "pragma_data_pack_count",
    "pragma_loop_flatten_count",
    "pragma_array_reshape_count",
    "pragma_other_count",
]

FEATURES_B: List[str] = FEATURES_A + FEATURES_B_ONLY

FEATURES_C_ONLY: List[str] = [
    "code_line_count",
    "code_non_blank_lines",
    "code_comment_lines",
    "code_function_count",
    "code_for_loop_count",
    "code_while_loop_count",
    "code_total_loop_count",
    "code_if_branch_count",
    "code_switch_branch_count",
    "code_max_nesting_depth",
    "code_arith_op_count",
    "code_bitwise_op_count",
    "code_array_subscript_count",
    "code_pointer_deref_count",
]

FEATURES_C: List[str] = FEATURES_B + FEATURES_C_ONLY

# Targets to Predict
TARGET_COLUMNS: List[str] = [
    "Worst-caseLatency",
    "LUT",
    "DSP",
    "FF",
    "BRAM_18K",
]


def _required_float(row: Dict[str, Any], field: str) -> float:
    """Read a required numeric field without silently manufacturing data."""
    if field not in row or row[field] is None:
        raise ValueError(f"Missing required numeric field: {field}")
    try:
        value = float(row[field])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {row[field]!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value for {field}: {value!r}")
    return value


def extract_features_from_record(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts all features (A, B, and C) from a single ForgeHLS design JSON record.
    Returns a dictionary mapping feature name to numeric float value.
    """
    feats: Dict[str, float] = {}

    # -------------------------------------------------------------
    # Feature Set A: Basic
    # -------------------------------------------------------------
    feats["TargetClockPeriod"] = _required_float(row, "TargetClockPeriod")
    feats["code_length"] = _required_float(row, "code_length")
    feats["pragma_number"] = _required_float(row, "pragma_number")

    # -------------------------------------------------------------
    # Extract Pragmas & Pre-process Source
    # -------------------------------------------------------------
    source_code_items = row.get("source_code")
    if not isinstance(source_code_items, list) or not source_code_items:
        raise ValueError("Missing or invalid required field: source_code")
    raw_texts = []
    
    # Initialize Pragma Counters
    pipeline_cnt = 0
    pipeline_off = 0
    pipeline_en = 0
    unroll_cnt = 0
    unroll_f1 = 0
    unroll_f2 = 0
    unroll_f4 = 0
    unroll_f8 = 0
    unroll_f16 = 0
    max_unroll_f = 0
    arr_part_cnt = 0
    arr_part_cyclic = 0
    arr_part_complete = 0
    arr_part_block = 0
    max_part_f = 0
    loop_tripcount_cnt = 0
    inline_cnt = 0
    interface_cnt = 0
    dataflow_cnt = 0
    stream_cnt = 0
    bind_storage_cnt = 0
    data_pack_cnt = 0
    loop_flatten_cnt = 0
    array_reshape_cnt = 0
    other_pragma_cnt = 0

    clean_code_lines = []

    for item in source_code_items:
        if not isinstance(item, dict):
            raise ValueError("Invalid source_code entry; expected an object")
        content = item.get("file_content", "")
        if not content:
            continue
        raw_texts.append(content)

        for line in content.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            match = PRAGMA_HLS_REGEX.search(line_str)
            if match:
                directive = match.group(1).upper()
                rem = match.group(2).strip().lower()

                if directive == "PIPELINE":
                    pipeline_cnt += 1
                    if "off" in rem:
                        pipeline_off += 1
                    else:
                        pipeline_en += 1

                elif directive == "UNROLL":
                    unroll_cnt += 1
                    factor_match = FACTOR_REGEX.search(rem)
                    factor = int(factor_match.group(1)) if factor_match else 0
                    if factor == 1:
                        unroll_f1 += 1
                    elif factor == 2:
                        unroll_f2 += 1
                    elif factor == 4:
                        unroll_f4 += 1
                    elif factor == 8:
                        unroll_f8 += 1
                    elif factor >= 16:
                        unroll_f16 += 1
                    max_unroll_f = max(max_unroll_f, factor)

                elif directive == "ARRAY_PARTITION":
                    arr_part_cnt += 1
                    if "cyclic" in rem:
                        arr_part_cyclic += 1
                    elif "complete" in rem:
                        arr_part_complete += 1
                    elif "block" in rem:
                        arr_part_block += 1
                    factor_match = FACTOR_REGEX.search(rem)
                    if factor_match:
                        max_part_f = max(max_part_f, int(factor_match.group(1)))

                elif directive == "LOOP_TRIPCOUNT":
                    loop_tripcount_cnt += 1
                elif directive == "INLINE":
                    inline_cnt += 1
                elif directive == "INTERFACE":
                    interface_cnt += 1
                elif directive == "DATAFLOW":
                    dataflow_cnt += 1
                elif directive == "STREAM":
                    stream_cnt += 1
                elif directive == "BIND_STORAGE":
                    bind_storage_cnt += 1
                elif directive == "DATA_PACK":
                    data_pack_cnt += 1
                elif directive == "LOOP_FLATTEN":
                    loop_flatten_cnt += 1
                elif directive == "ARRAY_RESHAPE":
                    array_reshape_cnt += 1
                else:
                    other_pragma_cnt += 1
            else:
                clean_code_lines.append(line_str)

    # -------------------------------------------------------------
    # Feature Set B: Pragma-Aware Values
    # -------------------------------------------------------------
    feats["pragma_pipeline_count"] = float(pipeline_cnt)
    feats["pragma_pipeline_off"] = float(pipeline_off)
    feats["pragma_pipeline_enable"] = float(pipeline_en)
    feats["pragma_unroll_count"] = float(unroll_cnt)
    feats["pragma_unroll_factor_1"] = float(unroll_f1)
    feats["pragma_unroll_factor_2"] = float(unroll_f2)
    feats["pragma_unroll_factor_4"] = float(unroll_f4)
    feats["pragma_unroll_factor_8"] = float(unroll_f8)
    feats["pragma_unroll_factor_16"] = float(unroll_f16)
    feats["pragma_max_unroll_factor"] = float(max_unroll_f)
    feats["pragma_array_partition_count"] = float(arr_part_cnt)
    feats["pragma_array_partition_cyclic"] = float(arr_part_cyclic)
    feats["pragma_array_partition_complete"] = float(arr_part_complete)
    feats["pragma_array_partition_block"] = float(arr_part_block)
    feats["pragma_max_partition_factor"] = float(max_part_f)
    feats["pragma_loop_tripcount_count"] = float(loop_tripcount_cnt)
    feats["pragma_inline_count"] = float(inline_cnt)
    feats["pragma_interface_count"] = float(interface_cnt)
    feats["pragma_dataflow_count"] = float(dataflow_cnt)
    feats["pragma_stream_count"] = float(stream_cnt)
    feats["pragma_bind_storage_count"] = float(bind_storage_cnt)
    feats["pragma_data_pack_count"] = float(data_pack_cnt)
    feats["pragma_loop_flatten_count"] = float(loop_flatten_cnt)
    feats["pragma_array_reshape_count"] = float(array_reshape_cnt)
    feats["pragma_other_count"] = float(other_pragma_cnt)

    # -------------------------------------------------------------
    # Feature Set C: Structural Code Metrics
    # -------------------------------------------------------------
    full_source = "\n".join(raw_texts)
    
    # Calculate comment and code lines
    comment_lines = 0
    non_blank_lines = len(clean_code_lines)
    total_lines = full_source.count("\n") + 1

    # Max nesting depth and operator counts
    current_depth = 0
    max_depth = 0
    arith_ops = 0
    bitwise_ops = 0
    array_subscripts = 0
    pointer_derefs = 0

    # Scan clean code without comments for operators
    source_no_comments = COMMENT_BLOCK_REGEX.sub("", full_source)
    source_no_comments = COMMENT_LINE_REGEX.sub("", source_no_comments)

    for char in source_no_comments:
        if char == "{":
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char == "}":
            if current_depth > 0:
                current_depth -= 1
        elif char in "+-*/%":
            arith_ops += 1
        elif char in "&|^~":
            bitwise_ops += 1
        elif char == "[":
            array_subscripts += 1

    pointer_derefs = source_no_comments.count("->") + source_no_comments.count("*")

    for line in full_source.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("//") or trimmed.startswith("/*") or trimmed.startswith("*"):
            comment_lines += 1

    for_loops = len(FOR_LOOP_REGEX.findall(source_no_comments))
    while_loops = len(WHILE_LOOP_REGEX.findall(source_no_comments))
    if_branches = len(IF_BRANCH_REGEX.findall(source_no_comments))
    switch_branches = len(SWITCH_BRANCH_REGEX.findall(source_no_comments))
    func_count = len(FUNCTION_DEF_REGEX.findall(source_no_comments))

    feats["code_line_count"] = float(total_lines)
    feats["code_non_blank_lines"] = float(non_blank_lines)
    feats["code_comment_lines"] = float(comment_lines)
    feats["code_function_count"] = float(func_count)
    feats["code_for_loop_count"] = float(for_loops)
    feats["code_while_loop_count"] = float(while_loops)
    feats["code_total_loop_count"] = float(for_loops + while_loops)
    feats["code_if_branch_count"] = float(if_branches)
    feats["code_switch_branch_count"] = float(switch_branches)
    feats["code_max_nesting_depth"] = float(max_depth)
    feats["code_arith_op_count"] = float(arith_ops)
    feats["code_bitwise_op_count"] = float(bitwise_ops)
    feats["code_array_subscript_count"] = float(array_subscripts)
    feats["code_pointer_deref_count"] = float(pointer_derefs)

    return feats


def extract_targets_from_record(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts the 5 QoR target variables from a design JSON record.
    """
    return {col: _required_float(row, col) for col in TARGET_COLUMNS}
