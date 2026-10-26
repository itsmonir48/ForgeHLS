"""
config.py

Centralized configuration and dynamic path management for ForgeHLS ML pipeline.
Ensures portability across different OS environments, drives, and directory locations.
"""

import os
from pathlib import Path

# Base directory: hls_ml/
SCRIPTS_DIR = Path(__file__).resolve().parent
HLS_ML_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = HLS_ML_DIR.parent

# Sub-directories
DATA_DIR = HLS_ML_DIR / "data"
RESULTS_DIR = HLS_ML_DIR / "results"
FIGURES_DIR = HLS_ML_DIR / "figures"
NOTEBOOKS_DIR = HLS_ML_DIR / "notebooks"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Raw dataset JSON path (supports environment variable override)
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "data_of_designs_forgehls.json"
ALT_RAW_PATH = PROJECT_ROOT.parent / "dataset" / "data_of_designs_forgehls.json"
if not DEFAULT_RAW_PATH.exists() and ALT_RAW_PATH.exists():
    DEFAULT_RAW_PATH = ALT_RAW_PATH
RAW_DATASET_PATH = Path(os.environ.get("FORGEHLS_DATA_PATH", str(DEFAULT_RAW_PATH)))

# Pipeline Constants
TARGET_NAMES = ["Worst-caseLatency", "LUT", "DSP", "FF", "BRAM_18K"]
SEEDS = [42, 43, 44, 45, 46]
COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
CALIBRATION_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
