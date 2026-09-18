"""Configuration for PV analysis experiments.

Keep experiment code separated from submission pipeline.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
RESULT_DIR = PROJECT_ROOT / "results" / "pv_analysis"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_VERSION = "v59_v36_075_v48_025_20260915"
FUSION_WEIGHT_V48 = 0.25
