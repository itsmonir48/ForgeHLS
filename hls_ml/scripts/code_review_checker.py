"""
code_review_checker.py
Comprehensive automated code review and linter for all hls_ml scripts.
"""

import os
import sys
import py_compile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

def check_all_scripts():
    print("=" * 70)
    print("Automated Code Quality & Syntax Revision")
    print("=" * 70)

    py_files = sorted(SCRIPTS_DIR.glob("*.py"))
    print(f"Auditing {len(py_files)} Python script files in {SCRIPTS_DIR}...\n")

    issues_found = 0

    for py_file in py_files:
        filename = py_file.name
        # 1. Syntax check
        try:
            py_compile.compile(str(py_file), doraise=True)
            status = "PASSED"
        except Exception as e:
            status = f"SYNTAX ERROR: {e}"
            issues_found += 1

        # 2. Check for hardcoded paths vs dynamic relative paths
        with py_file.open("r", encoding="utf-8") as f:
            content = f.read()

        warnings = []
        if "C:\\Users\\Nur Mohammad Naim\\ForgeHLS\\hls_ml" in content:
            warnings.append("Hardcoded hls_ml directory path (recommend dynamic Path(__file__))")
        
        # 3. Check for non-ascii characters in print statements that might cause Windows cp1252 crash
        for line_no, line in enumerate(content.splitlines(), start=1):
            if "print(" in line:
                for ch in line:
                    if ord(ch) > 127 and ch not in "±²³":
                        warnings.append(f"Non-ascii char '{ch}' (ord {ord(ch)}) in print at line {line_no}")
                        break

        print(f"[{status}] {filename:<30}")
        for w in warnings:
            print(f"   --> WARNING: {w}")

    print("\n" + "=" * 70)
    if issues_found == 0:
        print("Summary: 0 Syntax Errors found across all codebase scripts!")
    else:
        print(f"Summary: {issues_found} syntax issues found!")
    print("=" * 70)

if __name__ == "__main__":
    check_all_scripts()
