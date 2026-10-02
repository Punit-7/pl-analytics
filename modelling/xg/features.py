import numpy as np
import pandas as pd
from sqlalchemy import text

GOAL_X, GOAL_Y, GOAL_WIDTH = 120.0, 40.0, 8.0
CATEGORICAL = ["body_part", "shot_type", "play_pattern", "technique"]
NUMERIC = ["distance", "angle", "under_pressure", "first_time"]
# Columns that contain or reveal the outcome: never features.
BANNED = {"outcome", "is_goal", "statsbomb_xg"}


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    dx = GOAL_X - df["x"]
    dy = (df["y"] - GOAL_Y).abs()
    out = df.copy()
    out["distance"] = np.hypot(dx, dy)
    out["angle"] = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)
    return out


def build_matrix(shots: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Feature matrix. Pass `columns` from training so test data gets the same columns."""
    df = add_geometry(shots)
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    X[["under_pressure", "first_time"]] = X[["under_pressure", "first_time"]].astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    leaked = BANNED & set(X.columns)
    if leaked:
        raise ValueError(f"leakage: banned columns in features: {sorted(leaked)}")
    return X


def load_shots(engine, schema: str = "mart") -> pd.DataFrame:
    """Non-penalty shots with a 0/1 target."""
    df = pd.read_sql(text(f"SELECT * FROM {schema}.sb_shot WHERE shot_type <> 'Penalty'"), engine)
    df["is_goal"] = (df["outcome"] == "Goal").astype(int)
    return df