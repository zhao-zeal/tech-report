"""Generate V36/V48/Fusion prediction outputs for component analysis.

This script intentionally does not modify submission code.
It only extracts component predictions and prepares experiment artifacts.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "submission") not in sys.path:
    sys.path.insert(0, str(ROOT / "submission"))

from utils import hyx_pv_pipeline
from .config import RESULT_DIR, FUSION_WEIGHT_V48


OUTPUT_DIR = RESULT_DIR / "component_predictions"


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v36, v48 = hyx_pv_pipeline._components()

    pred_v36 = v36.predict_pv_model(save_output=False)
    pred_v48 = v48.predict_pv_model(save_output=False)

    pred_v36.to_csv(OUTPUT_DIR / "v36.csv", index=False)
    pred_v48.to_csv(OUTPUT_DIR / "v48.csv", index=False)

    fusion = pred_v36.copy()
    fusion["pred"] = (
        (1 - FUSION_WEIGHT_V48) * pred_v36["pred"].to_numpy()
        + FUSION_WEIGHT_V48 * pred_v48["pred"].to_numpy()
    )
    fusion.to_csv(OUTPUT_DIR / "fusion.csv", index=False)

    print("Saved component predictions:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
