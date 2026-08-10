"""
RT Correction — ported from the Streamlit "RT Correction" page.
Parabolic ramp-profile fit with a time-block lookup table.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution

DEFAULT_PARAMS = {"w": 0.3, "n1": 29, "n2": 72, "b": 39}
DE_BOUNDS = [(0.3, 0.3), (5, 40), (55, 95), (35, 40)]


def _time_blocks():
    start = datetime.strptime("00:00", "%H:%M")
    return [
        f"{(start + timedelta(minutes=15 * i)).strftime('%H:%M')} - "
        f"{(start + timedelta(minutes=15 * (i + 1))).strftime('%H:%M')}"
        for i in range(96)
    ]


def _prep(actual_rows, trend_rows):
    n = 96
    actual = np.zeros(n)
    trend = np.zeros(n)
    actual[: min(n, len(actual_rows))] = actual_rows[:n]
    trend[: min(n, len(trend_rows))] = trend_rows[:n]

    df = pd.DataFrame({"Actual": actual, "Trend": trend})
    df["Time-Blocks"] = _time_blocks()
    df["Blocks"] = np.arange(1, 97)
    return df


def _objective_factory(df, actual, trend, blocks, mask):
    def objective(x):
        w, n1, n2, b = x
        n1, n2, b = int(round(n1)), int(round(n2)), int(round(b))
        if not (n1 < b < n2):
            return 1e6

        p = df.loc[df["Blocks"].isin([b - 1, b, b + 1]), "Actual"].mean()
        calc = p * (((n1 - blocks) * (n2 - blocks)) / ((n1 - b) * (n2 - b)))
        projection = np.where(calc < 0, 0, calc)

        pred = projection[mask]
        act = actual[mask]

        block_error = np.mean(np.abs(act - pred)) / act.max()
        peak_error = abs(act.max() - pred.max()) / act.max()
        energy_error = abs(act.sum() - pred.sum()) / act.sum()
        return 0.80 * block_error + 0.10 * peak_error + 0.10 * energy_error

    return objective


def _final_calc(df, params):
    actual = df["Actual"].to_numpy(dtype=float)
    trend = df["Trend"].to_numpy(dtype=float)
    blocks = df["Blocks"].to_numpy(dtype=float)

    w, n1, n2, b = params["w"], params["n1"], params["n2"], params["b"]

    p = df.loc[df["Blocks"].isin([b - 1, b, b + 1]), "Actual"].mean()
    calc = p * (((n1 - blocks) * (n2 - blocks)) / ((n1 - b) * (n2 - b)))
    projection = np.where(calc < 0, 0, calc)
    df["Projection"] = projection

    lookup_blocks = [n1, n2, n1 + 3, n2 - 3]
    lookup_names = [
        "Parabolic Power Generation Starting Block",
        "Parabolic Power Generation Ending Block",
        "Actual Generation Available Block (Lower Limit)",
        "Actual Generation Effective Block (Upper Limit)",
    ]
    lookup_df = pd.DataFrame({"Parameter": lookup_names, "Block": lookup_blocks})
    lookup_df["Time Block"] = lookup_df["Block"].map(df.set_index("Blocks")["Time-Blocks"])

    return {
        "params": params,
        "chart": {
            "x": df["Blocks"].tolist(),
            "projection": np.round(projection, 4).tolist(),
            "actual": np.round(actual, 4).tolist(),
        },
        "time_block_lookup": lookup_df.to_dict(orient="records"),
    }


def optimize(actual_rows, trend_rows, max_iter=100) -> dict:
    df = _prep(actual_rows, trend_rows)
    actual = df["Actual"].to_numpy(dtype=float)
    trend = df["Trend"].to_numpy(dtype=float)
    blocks = df["Blocks"].to_numpy(dtype=float)
    mask = actual > 0.5

    objective = _objective_factory(df, actual, trend, blocks, mask)
    result = differential_evolution(
        objective, bounds=DE_BOUNDS, popsize=20, maxiter=max_iter, polish=True, seed=42,
    )
    w, n1, n2, b = result.x
    params = {"w": float(w), "n1": int(round(n1)), "n2": int(round(n2)), "b": int(round(b))}

    out = _final_calc(df, params)
    out["optimizer_score"] = float(result.fun)
    return out


def recalculate(actual_rows, trend_rows, params: dict) -> dict:
    df = _prep(actual_rows, trend_rows)
    return _final_calc(df, params)
