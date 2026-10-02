"""Train the xG models: two baselines, logistic regression and LightGBM."""
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from modelling.metrics import binary_report
from modelling.xg.features import build_matrix, load_shots

log = logging.getLogger("modelling.xg.train")
ART = ROOT / "modelling" / "artifacts"
REP = ROOT / "modelling" / "reports"


def group_split(df: pd.DataFrame, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["match_id"]))
    train, test = df.iloc[tr], df.iloc[te]
    overlap = set(train["match_id"]) & set(test["match_id"])
    if overlap:
        raise RuntimeError(f"leakage: {len(overlap)} matches in both sets")
    return train, test


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    m, seed = s.modelling, s.modelling["random_seed"]
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    shots = load_shots(make_engine())
    log.info("%d non-penalty shots, goal rate %.3f", len(shots), shots["is_goal"].mean())
    train, test = group_split(shots, m["xg_test_size"], seed)
    X_tr = build_matrix(train)
    X_te = build_matrix(test, columns=X_tr.columns)
    y_tr, y_te = train["is_goal"].to_numpy(), test["is_goal"].to_numpy()
    log.info("train %d / test %d shots, %d features", len(y_tr), len(y_te), X_tr.shape[1])

    preds = {}
    # Baseline 1: every shot gets the training goal rate.
    preds["baseline_constant"] = np.full(len(y_te), y_tr.mean())
    # Baseline 2: logistic regression on distance only.
    b2 = LogisticRegression(max_iter=1000).fit(X_tr[["distance"]], y_tr)
    preds["baseline_distance"] = b2.predict_proba(X_te[["distance"]])[:, 1]
    # Model 1: logistic regression on all features, standardised.
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logit.fit(X_tr, y_tr)
    preds["logistic"] = logit.predict_proba(X_te)[:, 1]
    # Model 2: LightGBM with early stopping on a validation split of the training data.
    inner_tr, inner_va = group_split(train, 0.2, seed + 1)
    gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=15,
                             min_child_samples=100, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=seed, verbose=-1)
    gbm.fit(X_tr.loc[inner_tr.index], inner_tr["is_goal"],
            eval_set=[(X_tr.loc[inner_va.index], inner_va["is_goal"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    preds["lightgbm"] = gbm.predict_proba(X_te)[:, 1]
    # Reference only, never a feature: StatsBomb's own xG.
    preds["reference_statsbomb"] = test["statsbomb_xg"].to_numpy()

    report = {name: binary_report(y_te, p) for name, p in preds.items()}
    for name, r in report.items():
        log.info("%-20s log_loss %.4f  brier %.4f  auc %.3f  xG %.0f vs goals %d",
                 name, r["log_loss"], r["brier"], r["auc"], r["xg_sum"], r["goals"])
    log.info("LightGBM stopped at %d trees", gbm.best_iteration_)

    joblib.dump(logit, ART / "xg_logistic.joblib")
    joblib.dump(gbm, ART / "xg_lightgbm.joblib")
    X_te.to_csv(ART / "xg_X_test.csv", index=False)
    out = pd.DataFrame({"match_id": test["match_id"].to_numpy(),
                        "competition_id": test["competition_id"].to_numpy(),
                        "y": y_te, **{f"p_{k}": v for k, v in preds.items()}})
    out.to_csv(REP / "xg_test_predictions.csv", index=False)
    (REP / "xg_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()