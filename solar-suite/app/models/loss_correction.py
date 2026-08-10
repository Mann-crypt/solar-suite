"""
Loss Correction — ported from the Streamlit "Loss Correction" page.

Every numeric operation below is copied verbatim from the original
Streamlit branches (cluster/non-cluster x Fixed/Tracking). Only the
Streamlit UI calls (st.*, session_state) have been removed and replaced
with plain function args/returns so this can run behind FastAPI.

Public entry points:
    detect_workbook(file_bytes) -> dict(is_cluster, ghi_cols, input_rows)
    run_fixed(file_bytes, plant_type, edited_rows=None) -> dict
    optimize_tracking(file_bytes, plant_type, edited_rows=None) -> dict (runs DE, slow)
    recalculate_tracking(file_bytes, plant_type, params, edited_rows=None) -> dict (fast, no DE)
"""

import io
import random
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

GHI_COLS_CLUSTER = ["CL1-GHI", "CL2-GHI", "CL3-GHI", "CL4-GHI", "CL5-GHI"]
GHI_COLS_SINGLE = ["GHI_Forecast"]

DE_BOUNDS = [
    (0, 10),    # DHI (%)
    (0, 30),    # GHI Starting Block
    (65, 80),   # GHI Ending Block
    (44, 60),   # GHI Max Block
    (0, 70),    # Tracking East Limit
    (0, 70),    # Tracking West Limit
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _xls(file_bytes: bytes) -> pd.ExcelFile:
    return pd.ExcelFile(io.BytesIO(file_bytes))


def detect_workbook(file_bytes: bytes) -> dict:
    xls = _xls(file_bytes)
    is_cluster = "Fixed-CL1" in xls.sheet_names
    sheet = "Fixed-CL1" if is_cluster else "Fixed"
    ghi_cols = GHI_COLS_CLUSTER if is_cluster else GHI_COLS_SINGLE

    df_fix = xls.parse(sheet_name=sheet, header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    df_fix["Actual"] = df_fix["Actual"].fillna(0)

    null_indices = df_fix[df_fix["Date"].isna()].index
    if len(null_indices) > 0:
        first_null = df_fix.index.get_loc(null_indices[0])
        df_fix = df_fix.iloc[:first_null]
    df_fix = df_fix.iloc[:96].copy()

    input_df = df_fix[ghi_cols + ["Actual"]].copy()
    input_df[ghi_cols] = input_df[ghi_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    input_df["Actual"] = pd.to_numeric(input_df["Actual"], errors="coerce").fillna(0)

    return {
        "is_cluster": is_cluster,
        "ghi_cols": ghi_cols,
        "rows": input_df.to_dict(orient="records"),
    }


def _apply_edits(df_fix: pd.DataFrame, ghi_cols, edited_rows):
    """edited_rows: list of dicts matching ghi_cols + Actual, len<=96, in order."""
    if not edited_rows:
        return df_fix
    edited = pd.DataFrame(edited_rows)
    edited[ghi_cols] = edited[ghi_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    edited["Actual"] = pd.to_numeric(edited["Actual"], errors="coerce").fillna(0)
    edited = edited.iloc[:96].reset_index(drop=True)
    n = min(len(edited), len(df_fix))
    df_fix.loc[:n - 1, ghi_cols] = edited[ghi_cols].values[:n]
    df_fix.loc[:n - 1, "Actual"] = edited["Actual"].values[:n]
    return df_fix


def _solar_geometry_cluster(df_fix, lat, month_lookup=None):
    """Populates declination/elevation/POA columns for CL1..CL5 (Fixed tilt version)."""
    df_fix["Date"] = pd.Timestamp.today()
    first_date = pd.Timestamp.today().replace(month=1, day=1).normalize()

    df_fix["Declination Angle ∆"] = 23.45 * (
        np.sin(np.radians(360 * (284 + (df_fix["Date"] - first_date).dt.days + 1) / 365))
    )
    df_fix["Elevation angle a"] = 90 - lat + df_fix["Declination Angle ∆"]

    if month_lookup is not None:
        df_fix["Tilt Angle b"] = df_fix["Date"].dt.strftime("%B").map(month_lookup)
    else:
        df_fix["Tilt Angle b"] = 0

    df_fix["a+b"] = df_fix["Elevation angle a"] + df_fix["Tilt Angle b"]
    df_fix["SIN(a+b)"] = np.sin(np.radians(df_fix["a+b"]))
    df_fix["Sin(a)"] = np.sin(np.radians(df_fix["Elevation angle a"]))

    for cl in ["CL1", "CL2", "CL3", "CL4", "CL5"]:
        ghi_col = f"{cl}-GHI"
        suffix = "" if cl == "CL1" else f"-{cl}"
        df_fix[f"GHI*sin(a){suffix}"] = df_fix[ghi_col] * df_fix["Sin(a)"]
        df_fix[f"GHI*sin(a+b){suffix}"] = df_fix[ghi_col] * df_fix["SIN(a+b)"]
        df_fix[f"POA fixed{suffix}"] = df_fix[f"GHI*sin(a+b){suffix}"] / df_fix["Sin(a)"]

    return df_fix


def _read_area_efficiency(xls):
    df = xls.parse(sheet_name="Area & Efficiency", header=[1], usecols=range(8))
    null_indices = df[df["Module Type"].isna()].index
    first_null_pos = df.index.get_loc(null_indices[0])
    df = df.iloc[:first_null_pos]
    df.columns = df.columns.str.strip()
    return df


def _chart_payload(x_label, forecast, actual):
    return {
        "x": list(range(1, len(actual) + 1)),
        "forecast": np.round(np.asarray(forecast, dtype=float), 4).tolist(),
        "actual": np.round(np.asarray(actual, dtype=float), 4).tolist(),
    }


def _display_table(df):
    display_df = df[
        ["Module Type", "Standard PV Efficiency (%)", "Efficiency Losses(%)", "Net Efficiency (%)", "Total area(m2)"]
    ].copy()
    num_cols = display_df.select_dtypes(include="number").columns
    display_df[num_cols] = display_df[num_cols].round(2)
    return display_df.to_dict(orient="records")


# --------------------------------------------------------------------------
# CLUSTER + FIXED
# --------------------------------------------------------------------------

def _run_cluster_fixed(xls, df_fix, ghi_cols):
    df = _read_area_efficiency(xls)
    df_w = xls.parse(sheet_name="Area & Efficiency", header=2, usecols=[12, 13, 14, 15, 16])
    df_st = xls.parse(sheet_name="Forecast Config", header=[8])
    lat = float(df_st.loc[0, "Lat"])

    df_tilt = xls.parse(sheet_name="Config Tilt Angle", header=[7])
    df_tilt.columns = df_tilt.columns.str.strip()
    null_indices = df_tilt["Fixed"].isna()
    null_indices = df_tilt[null_indices].index
    first_null_pos = df_tilt.index.get_loc(null_indices[0])
    df_tilt = df_tilt.iloc[:first_null_pos]
    df_tilt = df_tilt.dropna(how="all", axis=1)
    df_tilt = df_tilt.rename(columns={"Unnamed: 2": "Month_Num", "Unnamed: 3": "Month"})
    month_lookup = df_tilt.set_index("Month")["Fixed"].to_dict()

    df_fix = _solar_geometry_cluster(df_fix, lat, month_lookup)

    max_loss = df["Standard PV Efficiency (%)"].min()
    results = []

    def compute_total_power(loss):
        df["Efficiency Losses(%)"] = loss
        df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
        df_weight = pd.DataFrame({
            f"CL-{i}": ((df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100) * df_w[f"CL-{i}"].values[0:1]
            for i in range(1, 6)
        })
        total = 0
        for i, suffix in zip(range(1, 6), ["", "-CL2", "-CL3", "-CL4", "-CL5"]):
            power = (df_fix[f"POA fixed{suffix}"] * np.sum(df_weight[f"CL-{i}"])) / 1_000_000
            df_fix[f"CL{i}_Fixed Power=I*Ƞ*A"] = power
            total = total + power
        df_fix["Total Power (CL1+CL2+…)"] = total
        return df_weight

    for loss in np.arange(0, max_loss + 0.01, 0.1):
        compute_total_power(loss)
        actual_peak = df_fix["Actual"].max()
        predicted_peak = df_fix["Total Power (CL1+CL2+…)"].max()
        results.append({
            "Efficiency Loss (%)": loss,
            "Actual Peak": actual_peak,
            "Predicted Peak": predicted_peak,
            "Peak Error": abs(actual_peak - predicted_peak),
        })

    results_df = pd.DataFrame(results)
    best_loss = results_df.loc[results_df["Peak Error"].idxmin(), "Efficiency Loss (%)"]
    compute_total_power(best_loss)

    return {
        "is_cluster": True,
        "plant_type": "Fixed",
        "best_loss": float(best_loss),
        "chart": _chart_payload("Block", df_fix["Total Power (CL1+CL2+…)"], df_fix["Actual"]),
        "efficiency_table": _display_table(df),
    }


# --------------------------------------------------------------------------
# CLUSTER + TRACKING
# --------------------------------------------------------------------------

def _cluster_tracking_objective_factory(backend_list, ghi_arrays, weight_sum, actual, mask):
    blocks = backend_list[0]["Block No."]

    def objective(x):
        try:
            DHI, s, e, m, east, west = (int(round(v)) for v in x)
            if s >= m or m >= e:
                return 1e9

            m1 = 90 / (s - 1 - m)
            m2 = 90 / (e + 1 - m)

            zenith = np.where(
                blocks <= m,
                np.minimum(89, m1 * (blocks - m)),
                np.minimum(89, m2 * (blocks - m)),
            )
            panel = np.where(
                blocks < m,
                np.minimum(zenith, abs(east)),
                np.where((blocks > m) & (zenith > west), west, zenith),
            )
            cos_alpha = np.clip(np.cos(np.radians(panel)), 1e-6, None)

            prediction = np.zeros_like(blocks, dtype=np.float64)
            for i, ghi in enumerate(ghi_arrays):
                dhi = ghi * DHI / 100
                dni = (ghi - dhi) / cos_alpha
                prediction = prediction + (dni * weight_sum[i]) / 1_000_000

            prediction = prediction[mask]
            if np.isnan(prediction).any() or np.isinf(prediction).any() or actual.max() == 0:
                return 1e9

            block_error = np.mean(np.abs(actual - prediction)) / actual.max()
            peak_error = abs(actual.max() - prediction.max()) / actual.max()
            energy_error = abs(actual.sum() - prediction.sum()) / actual.sum()
            return 0.80 * block_error + 0.10 * peak_error + 0.10 * energy_error
        except Exception:
            return 1e9

    return objective


def _prep_cluster_tracking(xls, df_fix, ghi_cols):
    df = _read_area_efficiency(xls)
    df_w = xls.parse(sheet_name="Area & Efficiency", header=2, usecols=[12, 13, 14, 15, 16])
    df_st = xls.parse(sheet_name="Forecast Config", header=[8])
    lat = float(df_st.loc[0, "Lat"])

    df_fix = _solar_geometry_cluster(df_fix, lat, month_lookup=None)  # tracking: tilt = 0

    max_loss = df["Standard PV Efficiency (%)"].min()
    results = []

    def weight_for_loss(loss):
        df["Efficiency Losses(%)"] = loss
        df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
        return pd.DataFrame({
            f"CL-{i}": ((df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100) * df_w[f"CL-{i}"].values[0:1]
            for i in range(1, 6)
        })

    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df_weight = weight_for_loss(loss)
        total = 0
        for i, suffix in zip(range(1, 6), ["", "-CL2", "-CL3", "-CL4", "-CL5"]):
            total = total + (df_fix[f"POA fixed{suffix}"] * np.sum(df_weight[f"CL-{i}"])) / 1_000_000
        actual_peak = df_fix["Actual"].max()
        predicted_peak = total.max()
        results.append({"Efficiency Loss (%)": loss, "Peak Error": abs(actual_peak - predicted_peak)})

    results_df = pd.DataFrame(results)
    best_loss = results_df.loc[results_df["Peak Error"].idxmin(), "Efficiency Loss (%)"]
    df_weight = weight_for_loss(best_loss)

    df_bcal = [xls.parse(sheet_name=f"Backend Cal CL{i}") for i in range(1, 6)]
    df_trac = xls.parse(sheet_name="Tracking", header=[1])

    actual = df_fix["Actual"].to_numpy(dtype=np.float64)
    mask = actual != 0
    actual_masked = actual[mask]

    ghi_arrays = [df_fix[col].to_numpy(dtype=np.float64) for col in ghi_cols]
    weight_sum = np.array([df_weight[f"CL-{i}"].sum() for i in range(1, 6)], dtype=np.float64)

    return {
        "df": df, "df_w": df_w, "df_fix": df_fix, "df_weight": df_weight,
        "df_bcal": df_bcal, "df_trac": df_trac, "ghi_cols": ghi_cols,
        "best_loss": float(best_loss), "actual_masked": actual_masked, "mask": mask,
        "ghi_arrays": ghi_arrays, "weight_sum": weight_sum,
    }


def _finalize_cluster_tracking(prep, params):
    df, df_fix, df_bcal, df_trac = prep["df"], prep["df_fix"], prep["df_bcal"], prep["df_trac"]
    df_w = prep["df_w"]
    best_loss = params["loss"]
    DHI, s, e, m, east, west = params["dhi"], params["start"], params["end"], params["max"], params["east"], params["west"]

    df["Efficiency Losses(%)"] = best_loss
    df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
    df_weight = pd.DataFrame({
        f"CL-{i}": ((df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100) * df_w[f"CL-{i}"].values[0:1]
        for i in range(1, 6)
    })
    weights = df_weight.sum()

    m1 = 90 / (s - 1 - m)
    m2 = 90 / (e + 1 - m)
    blocks = df_bcal[0]["Block No."]

    zenith = np.where(blocks <= m, np.minimum(89, m1 * (blocks - m)), np.minimum(89, m2 * (blocks - m)))
    panel = np.where(blocks < m, np.minimum(zenith, abs(east)),
                      np.where((blocks > m) & (zenith > west), west, zenith))
    cos_alpha = np.clip(np.cos(np.radians(panel)), 1e-6, None)

    forecast = np.zeros(len(df_fix))
    for i, col in enumerate(prep["ghi_cols"], start=1):
        ghi = df_fix[col].to_numpy()
        dhi = ghi * DHI / 100
        dni = (ghi - dhi) / cos_alpha
        forecast = forecast + (dni * weights[f"CL-{i}"]) / 1_000_000

    df_trac["Fixed Power=I*Ƞ*A"] = forecast

    return {
        "is_cluster": True,
        "plant_type": "Tracking",
        "params": params,
        "chart": _chart_payload("Block", df_trac["Fixed Power=I*Ƞ*A"], df_fix["Actual"]),
        "efficiency_table": _display_table(df),
    }


# --------------------------------------------------------------------------
# NON-CLUSTER (single GHI column) + FIXED
# --------------------------------------------------------------------------

def _run_noncluster_fixed(xls, df_fix_edited, ghi_cols):
    df = xls.parse(sheet_name="Area & Efficiency", header=[1])
    df.columns = df.columns.str.strip()
    null_indices = df[df["Module Type"].isna()].index
    first_null_pos = df.index.get_loc(null_indices[0])
    df = df.iloc[:first_null_pos]

    df_st = xls.parse(sheet_name="Forecast Config", header=[8])
    lat = float(df_st.loc[0, "Lat"])

    df_tilt = xls.parse(sheet_name="Config Tilt Angle", header=[7])
    df_tilt.columns = df_tilt.columns.str.strip()
    df_tilt["Fixed"] = df_tilt["Fixed"].fillna(0)
    null_indices = df_tilt[df_tilt["Fixed"].isna()].index
    df_tilt = df_tilt.dropna(how="all", axis=1)
    df_tilt = df_tilt.rename(columns={"Unnamed: 2": "Month_Num", "Unnamed: 3": "Month"})
    month_lookup = df_tilt.set_index("Month")["Fixed"].to_dict() if "Month" in df_tilt.columns else {}

    df_fix = xls.parse(sheet_name="Fixed", header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    df_fix["GHI_Forecast"] = df_fix_edited["GHI_Forecast"].values[:len(df_fix)]
    df_fix["Actual"] = df_fix_edited["Actual"].values[:len(df_fix)]
    null_indices = df_fix[df_fix["Date"].isna()].index
    first_null_pos = df_fix.index.get_loc(null_indices[0])
    df_fix = df_fix.iloc[:first_null_pos]

    df_fix["Date"] = pd.Timestamp.today()
    first_date = pd.Timestamp.today().replace(month=1, day=1).normalize()
    df_fix["Declination Angle ∆"] = 23.45 * (
        np.sin(np.radians(360 * (284 + (df_fix["Date"] - first_date).dt.days + 1) / 365))
    )
    df_fix["Elevation angle a"] = 90 - lat + df_fix["Declination Angle ∆"]
    df_fix["Tilt Angle b"] = df_fix["Date"].dt.strftime("%B").map(month_lookup)
    df_fix["a+b"] = df_fix["Elevation angle a"] + df_fix["Tilt Angle b"]
    df_fix["SIN(a+b)"] = np.sin(np.radians(df_fix["a+b"]))
    df_fix["Sin(a)"] = np.sin(np.radians(df_fix["Elevation angle a"]))
    df_fix["GHI*sin(a)"] = df_fix["GHI_Forecast"] * df_fix["Sin(a)"]
    df_fix["GHI*sin(a+b)"] = df_fix["GHI_Forecast"] * df_fix["SIN(a+b)"]
    df_fix["POA fixed"] = df_fix["GHI*sin(a+b)"] / df_fix["Sin(a)"]

    max_loss = df["Standard PV Efficiency (%)"].min()
    results = []
    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df["Efficiency Losses(%)"] = loss
        df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
        df["Eff Area"] = (df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100
        df_fix["Fixed Power=I*Ƞ*A"] = (df_fix["POA fixed"] * np.sum(df["Eff Area"])) / 1_000_000
        actual_peak = df_fix["Actual"].max()
        predicted_peak = df_fix["Fixed Power=I*Ƞ*A"].max()
        results.append({"Efficiency Loss (%)": loss, "Peak Error": abs(actual_peak - predicted_peak)})

    results_df = pd.DataFrame(results)
    best_loss = results_df.loc[results_df["Peak Error"].idxmin(), "Efficiency Loss (%)"]

    df["Efficiency Losses(%)"] = best_loss
    df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
    df["Eff Area"] = (df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100
    df_fix["Fixed Power=I*Ƞ*A"] = (df_fix["POA fixed"] * df["Eff Area"].sum()) / 1_000_000

    return {
        "is_cluster": False,
        "plant_type": "Fixed",
        "best_loss": float(best_loss),
        "chart": _chart_payload("Block", df_fix["Fixed Power=I*Ƞ*A"], df_fix["Actual"]),
        "efficiency_table": _display_table(df),
    }


# --------------------------------------------------------------------------
# NON-CLUSTER (single GHI column) + TRACKING
# --------------------------------------------------------------------------

def _noncluster_tracking_objective_factory(blocks, ghi, eff_area, df_fix_actual):
    mask = df_fix_actual != 0
    actual = df_fix_actual[mask]

    def objective(x):
        DHI, s, e, m, east, west = (int(round(v)) for v in x)
        if s >= m or m >= e:
            return 1e9

        m1 = 90 / (s - 1 - m)
        m2 = 90 / (e + 1 - m)

        dhi = ghi * DHI / 100
        g_minus_d = ghi - dhi

        zenith = np.where(blocks <= m, np.minimum(89, m1 * (blocks - m)), np.minimum(89, m2 * (blocks - m)))
        panel = np.where(blocks < m, np.minimum(zenith, abs(east)),
                          np.where((blocks > m) & (zenith > west), west, zenith))
        cos_alpha = np.cos(np.radians(panel))

        dni = g_minus_d / cos_alpha
        prediction = (dni * eff_area) / 1_000_000
        prediction = prediction[mask]

        block_error = np.mean(np.abs(actual - prediction)) / actual.max()
        peak_error = abs(actual.max() - prediction.max()) / actual.max()
        energy_error = abs(actual.sum() - prediction.sum()) / actual.sum()
        return 0.80 * block_error + 0.10 * peak_error + 0.10 * energy_error

    return objective


def _prep_noncluster_tracking(xls, df_fix_edited, ghi_cols):
    df = xls.parse(sheet_name="Area & Efficiency", header=[1])
    df.columns = df.columns.str.strip()
    null_indices = df[df["Module Type"].isna()].index
    first_null_pos = df.index.get_loc(null_indices[0])
    df = df.iloc[:first_null_pos]

    df_st = xls.parse(sheet_name="Forecast Config", header=[8])
    lat = float(df_st.loc[0, "Lat"])

    df_fix = xls.parse(sheet_name="Fixed", header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    df_fix["GHI_Forecast"] = df_fix_edited["GHI_Forecast"].values[:len(df_fix)]
    df_fix["Actual"] = df_fix_edited["Actual"].values[:len(df_fix)]
    df_fix["Actual"] = df_fix["Actual"].fillna(0)
    null_indices = df_fix[df_fix["Date"].isna()].index
    first_null_pos = df_fix.index.get_loc(null_indices[0])
    df_fix = df_fix.iloc[:first_null_pos]

    df_fix["Date"] = pd.Timestamp.today()
    first_date = pd.Timestamp.today().replace(month=1, day=1).normalize()
    df_fix["Declination Angle ∆"] = 23.45 * (
        np.sin(np.radians(360 * (284 + (df_fix["Date"] - first_date).dt.days + 1) / 365))
    )
    df_fix["Elevation angle a"] = 90 - lat + df_fix["Declination Angle ∆"]
    df_fix["Tilt Angle b"] = 0
    df_fix["a+b"] = df_fix["Elevation angle a"] + df_fix["Tilt Angle b"]
    df_fix["SIN(a+b)"] = np.sin(np.radians(df_fix["a+b"]))
    df_fix["Sin(a)"] = np.sin(np.radians(df_fix["Elevation angle a"]))
    df_fix["GHI*sin(a)"] = df_fix["GHI_Forecast"] * df_fix["Sin(a)"]
    df_fix["GHI*sin(a+b)"] = df_fix["GHI_Forecast"] * df_fix["SIN(a+b)"]
    df_fix["POA fixed"] = df_fix["GHI*sin(a+b)"] / df_fix["Sin(a)"]

    max_loss = df["Standard PV Efficiency (%)"].min()
    results = []
    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df["Efficiency Losses(%)"] = loss
        df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
        df["Eff Area"] = (df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100
        df_fix["Fixed Power=I*Ƞ*A"] = (df_fix["POA fixed"] * np.sum(df["Eff Area"])) / 1_000_000
        actual_peak = df_fix["Actual"].max()
        predicted_peak = df_fix["Fixed Power=I*Ƞ*A"].max()
        results.append({"Efficiency Loss (%)": loss, "Peak Error": abs(actual_peak - predicted_peak)})

    results_df = pd.DataFrame(results)
    best_loss = results_df.loc[results_df["Peak Error"].idxmin(), "Efficiency Loss (%)"]
    df["Efficiency Losses(%)"] = best_loss
    df["Net Efficiency (%)"] = df["Standard PV Efficiency (%)"] - df["Efficiency Losses(%)"]
    df["Eff Area"] = (df["Total area(m2)"] * df["Net Efficiency (%)"]) / 100

    df_bcal = xls.parse(sheet_name="Backend Cal")
    df_trac = xls.parse(sheet_name="Tracking", header=[1])

    blocks = df_bcal["Block No."].to_numpy(dtype=np.float64)
    ghi = df_fix["GHI_Forecast"].to_numpy(dtype=np.float64)
    eff_area = df["Eff Area"].sum()
    actual = df_fix["Actual"].to_numpy(dtype=np.float64)

    return {
        "df": df, "df_fix": df_fix, "df_bcal": df_bcal, "df_trac": df_trac,
        "blocks": blocks, "ghi": ghi, "eff_area": eff_area, "actual": actual,
        "best_loss": float(best_loss),
    }


def _finalize_noncluster_tracking(prep, params):
    df_fix, df_trac = prep["df_fix"], prep["df_trac"]
    df = prep["df"]
    DHI, s, e, m, east, west = params["dhi"], params["start"], params["end"], params["max"], params["east"], params["west"]

    m1 = 90 / (s - 1 - m)
    m2 = 90 / (e + 1 - m)

    ghi = prep["ghi"]
    blocks = prep["blocks"]
    dhi = ghi * DHI / 100.0
    ghi_minus_dhi = ghi - dhi

    zenith = np.where(blocks <= m, np.minimum(89.0, m1 * (blocks - m)), np.minimum(89.0, m2 * (blocks - m)))
    panel = np.where(blocks < m, np.minimum(zenith, abs(east)),
                      np.where((blocks > m) & (zenith > west), west, zenith))
    cos_alpha = np.clip(np.cos(np.radians(panel)), 1e-6, None)

    dni = ghi_minus_dhi / cos_alpha
    forecast = dni * prep["eff_area"] / 1_000_000
    df_trac["Fixed Power=I*Ƞ*A"] = forecast

    return {
        "is_cluster": False,
        "plant_type": "Tracking",
        "params": params,
        "chart": _chart_payload("Block", df_trac["Fixed Power=I*Ƞ*A"], df_fix["Actual"]),
        "efficiency_table": _display_table(df),
    }


# --------------------------------------------------------------------------
# Public API used by FastAPI routes
# --------------------------------------------------------------------------

def run_fixed(file_bytes: bytes, edited_rows=None) -> dict:
    """Fixed plant type: no DE optimization needed, immediate result."""
    xls = _xls(file_bytes)
    is_cluster = "Fixed-CL1" in xls.sheet_names
    sheet = "Fixed-CL1" if is_cluster else "Fixed"
    ghi_cols = GHI_COLS_CLUSTER if is_cluster else GHI_COLS_SINGLE

    df_fix = xls.parse(sheet_name=sheet, header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    df_fix["Actual"] = df_fix["Actual"].fillna(0)
    null_indices = df_fix[df_fix["Date"].isna()].index
    if len(null_indices) > 0:
        first_null = df_fix.index.get_loc(null_indices[0])
        df_fix = df_fix.iloc[:first_null]
    df_fix = df_fix.iloc[:96].copy()
    df_fix = _apply_edits(df_fix, ghi_cols, edited_rows)

    if is_cluster:
        return _run_cluster_fixed(xls, df_fix, ghi_cols)
    else:
        return _run_noncluster_fixed(xls, df_fix, ghi_cols)


def optimize_tracking(file_bytes: bytes, edited_rows=None, max_iter=100) -> dict:
    """Tracking plant type: runs the scan + differential_evolution (slow ~seconds)."""
    xls = _xls(file_bytes)
    is_cluster = "Fixed-CL1" in xls.sheet_names
    sheet = "Fixed-CL1" if is_cluster else "Fixed"
    ghi_cols = GHI_COLS_CLUSTER if is_cluster else GHI_COLS_SINGLE

    df_fix = xls.parse(sheet_name=sheet, header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    df_fix["Actual"] = df_fix["Actual"].fillna(0)
    null_indices = df_fix[df_fix["Date"].isna()].index
    if len(null_indices) > 0:
        first_null = df_fix.index.get_loc(null_indices[0])
        df_fix = df_fix.iloc[:first_null]
    df_fix = df_fix.iloc[:96].copy()
    df_fix = _apply_edits(df_fix, ghi_cols, edited_rows)

    if is_cluster:
        prep = _prep_cluster_tracking(xls, df_fix, ghi_cols)
        objective = _cluster_tracking_objective_factory(
            prep["df_bcal"], prep["ghi_arrays"], prep["weight_sum"], prep["actual_masked"], prep["mask"]
        )
    else:
        prep = _prep_noncluster_tracking(xls, df_fix, ghi_cols)
        objective = _noncluster_tracking_objective_factory(
            prep["blocks"], prep["ghi"], prep["eff_area"], prep["actual"]
        )

    result = differential_evolution(
        objective, bounds=DE_BOUNDS, strategy="best1bin", maxiter=max_iter,
        popsize=15, tol=0.001, mutation=(0.5, 1), recombination=0.7,
        seed=42, polish=True, workers=1,
    )
    best = np.round(result.x).astype(int)
    params = {
        "loss": prep["best_loss"],
        "dhi": int(best[0]), "start": int(best[1]), "end": int(best[2]),
        "max": int(best[3]), "east": int(best[4]), "west": int(best[5]),
    }

    if is_cluster:
        out = _finalize_cluster_tracking(prep, params)
    else:
        out = _finalize_noncluster_tracking(prep, params)
    out["optimizer_score"] = float(result.fun)
    return out


def recalculate_tracking(file_bytes: bytes, params: dict, edited_rows=None) -> dict:
    """Fast path: user tweaked the optimized numbers, recompute the chart without rerunning DE."""
    xls = _xls(file_bytes)
    is_cluster = "Fixed-CL1" in xls.sheet_names
    sheet = "Fixed-CL1" if is_cluster else "Fixed"
    ghi_cols = GHI_COLS_CLUSTER if is_cluster else GHI_COLS_SINGLE

    df_fix = xls.parse(sheet_name=sheet, header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    df_fix["Actual"] = df_fix["Actual"].fillna(0)
    null_indices = df_fix[df_fix["Date"].isna()].index
    if len(null_indices) > 0:
        first_null = df_fix.index.get_loc(null_indices[0])
        df_fix = df_fix.iloc[:first_null]
    df_fix = df_fix.iloc[:96].copy()
    df_fix = _apply_edits(df_fix, ghi_cols, edited_rows)

    if is_cluster:
        prep = _prep_cluster_tracking(xls, df_fix, ghi_cols)
        return _finalize_cluster_tracking(prep, params)
    else:
        prep = _prep_noncluster_tracking(xls, df_fix, ghi_cols)
        return _finalize_noncluster_tracking(prep, params)
