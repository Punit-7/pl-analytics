"""Compare backtest forecasts with market-average closing odds on the same matches."""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from data.common.config import ROOT, load_settings  # noqa: E402
from data.common.db import make_engine  # noqa: E402
from data.common.logging_setup import setup_logging  # noqa: E402
from modelling.match.data import load_closing_odds  # noqa: E402
from modelling.metrics import multiclass_log_loss, rps  # noqa: E402

log = logging.getLogger("modelling.match.benchmark")
REP = ROOT / "modelling" / "reports"
ODDS = ["avg_close_home_odds", "avg_close_draw_odds", "avg_close_away_odds"]


def implied_probs(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proportional method: 1/odds, rescaled to sum to 1. Also returns the overround."""
    q = 1.0 / odds
    total = q.sum(axis=1, keepdims=True)
    return q / total, total.ravel()


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def calibration_plot(y: np.ndarray, dc: np.ndarray, book: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, k, name in ((axes[0], 0, "Home win"), (axes[1], 1, "Draw")):
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        for label, p in (("Dixon-Coles", dc[:, k]), ("Closing odds", book[:, k])):
            observed, predicted = calibration_curve(
                (y == k).astype(int), p, n_bins=10, strategy="quantile"
            )
            ax.plot(predicted, observed, marker="o", label=label)
        ax.set_title(f"{name} probability calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend()
    fig.savefig(REP / "figures" / "match_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    bt = pd.read_csv(REP / "backtest_predictions.csv")
    df = bt.merge(load_closing_odds(make_engine()), on="match_id", how="inner")
    y = df["outcome"].to_numpy()
    book, overround = implied_probs(df[ODDS].to_numpy())
    dc = df[["dc_h", "dc_d", "dc_a"]].to_numpy()

    df["dc_rps"], df["book_rps"] = rps(dc, y), rps(book, y)
    gap, low, high = paired_bootstrap(
        (df["dc_rps"] - df["book_rps"]).to_numpy(), seed=s.modelling["random_seed"]
    )
    by_season = df.groupby("season")[["dc_rps", "book_rps"]].mean()
    by_season["gap"] = by_season["dc_rps"] - by_season["book_rps"]
    calibration_plot(y, dc, book)

    summary = {
        "matches": int(len(df)),
        "mean_overround": float(overround.mean()),
        "dc_rps": float(df["dc_rps"].mean()),
        "book_rps": float(df["book_rps"].mean()),
        "rps_gap": gap,
        "rps_gap_95ci": [low, high],
        "dc_log_loss": multiclass_log_loss(dc, y),
        "book_log_loss": multiclass_log_loss(book, y),
        "by_season": by_season.round(4).to_dict(orient="index"),
    }
    (REP / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "%d matches | DC RPS %.4f | odds RPS %.4f | gap %.4f (95%% CI %.4f to %.4f)",
        len(df),
        summary["dc_rps"],
        summary["book_rps"],
        gap,
        low,
        high,
    )
    log.info("by season:\n%s", by_season.round(4).to_string())


if __name__ == "__main__":
    main()
