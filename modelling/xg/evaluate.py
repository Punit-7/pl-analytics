"""Calibration plots and SHAP explanations for the xG models."""

import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from data.common.config import ROOT, load_settings  # noqa: E402
from data.common.logging_setup import setup_logging  # noqa: E402
from modelling.metrics import expected_calibration_error  # noqa: E402

log = logging.getLogger("modelling.xg.evaluate")
ART = ROOT / "modelling" / "artifacts"
REP = ROOT / "modelling" / "reports"
FIG = REP / "figures"
MODELS = {
    "p_logistic": "Logistic regression",
    "p_lightgbm": "LightGBM",
    "p_reference_statsbomb": "StatsBomb xG (reference)",
}


def reliability_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 0.6], [0, 0.6], linestyle="--", color="grey", label="Perfect calibration")
    for col, label in MODELS.items():
        observed, predicted = calibration_curve(df["y"], df[col], n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.set_xlabel("Mean predicted xG in bin")
    ax.set_ylabel("Observed goal rate in bin")
    ax.set_title("xG calibration on test matches\nData: StatsBomb Open Data 2015/16")
    ax.legend()
    fig.savefig(FIG / "xg_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REP / "xg_test_predictions.csv")

    reliability_plot(df)
    ece = {col: expected_calibration_error(df["y"], df[col]) for col in MODELS}
    by_competition = {
        int(c): float(log_loss(g["y"], g["p_lightgbm"].clip(1e-6, 1 - 1e-6), labels=[0, 1]))
        for c, g in df.groupby("competition_id")
    }

    gbm = joblib.load(ART / "xg_lightgbm.joblib")
    X_te = pd.read_csv(ART / "xg_X_test.csv")
    sample = X_te.sample(min(5000, len(X_te)), random_state=s.modelling["random_seed"])
    values = shap.TreeExplainer(gbm).shap_values(sample)
    if isinstance(values, list):  # older SHAP versions return one array per class
        values = values[1]
    shap.summary_plot(values, sample, max_display=15, show=False)
    plt.savefig(FIG / "xg_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    shap.dependence_plot("distance", values, sample, show=False)
    plt.savefig(FIG / "xg_shap_distance.png", dpi=150, bbox_inches="tight")
    plt.close()
    importance = pd.Series(np.abs(values).mean(axis=0), index=sample.columns).sort_values(
        ascending=False
    )

    result = {
        "ece": ece,
        "lightgbm_log_loss_by_competition": by_competition,
        "shap_mean_abs_top15": importance.head(15).round(4).to_dict(),
    }
    (REP / "xg_evaluation.json").write_text(json.dumps(result, indent=2))
    for col, value in ece.items():
        log.info("ECE %-24s %.4f", MODELS[col], value)
    log.info("Top SHAP features: %s", ", ".join(importance.head(5).index))


if __name__ == "__main__":
    main()
