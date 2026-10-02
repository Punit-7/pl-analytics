import pandas as pd

from modelling.xg.features import BANNED, add_geometry, build_matrix


def fake_shots() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [108.0, 108.0], "y": [40.0, 30.0],
        "under_pressure": [True, False], "first_time": [False, True],
        "body_part": ["Right Foot", "Head"], "shot_type": ["Open Play", "Open Play"],
        "play_pattern": ["Regular Play", "From Corner"], "technique": ["Normal", "Normal"],
        "outcome": ["Goal", "Saved"], "statsbomb_xg": [0.3, 0.05],
    })


def test_no_banned_columns_in_features():
    X = build_matrix(fake_shots())
    assert not BANNED & set(X.columns)


def test_geometry_matches_hand_calculation():
    g = add_geometry(fake_shots())
    assert abs(g.loc[0, "distance"] - 12.0) < 1e-9
    assert abs(g.loc[0, "angle"] - 0.6435011) < 1e-6
    assert abs(g.loc[1, "distance"] - 15.6204994) < 1e-6
    assert abs(g.loc[1, "angle"] - 0.3985224) < 1e-6


def test_test_matrix_gets_training_columns():
    X_train = build_matrix(fake_shots())
    X_test = build_matrix(fake_shots().iloc[[0]], columns=X_train.columns)
    assert list(X_test.columns) == list(X_train.columns)