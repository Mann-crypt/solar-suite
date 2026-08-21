# ============================================================
# SOLAR FORECAST CORRECTION
# COMPACT CALCULATION ENGINE
# FIXED / TRACKING
# ============================================================

import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# CONSTANTS
# ============================================================

PLANTS = ["Fixed", "Tracking"]

CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

POA_COLS = [
    "POA fixed",
    "POA Fixed-C12",
    "POA Fixed-C13",
    "POA Fixed-C14",
    "POA Fixed-C15",
]

POWER_COLS = [
    f"CL{i}_Fixed Power=I*Ƞ*A"
    for i in range(1, 6)
]

TOTAL_POWER = "Total Power (CL1+CL2+…)"


TRACKING_BOUNDS = [
    (0, 10),      # DHI
    (10, 30),     # GHI Starting Block
    (65, 80),     # GHI Ending Block
    (47, 53),     # GHI Max Block
    (10, 70),     # East
    (10, 70),     # West
]


# ============================================================
# BASIC HELPERS
# ============================================================

def num_series(x):
    return pd.to_numeric(x, errors="coerce").fillna(0)


def num_array(x):
    return num_series(x).to_numpy(dtype=float)


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def ensure_df(x, name="DataFrame"):
    if not isinstance(x, pd.DataFrame):
        raise TypeError(
            f"{name} must be a DataFrame, "
            f"got {type(x).__name__}"
        )
    return x.copy()


def read_excel(data, sheet, **kwargs):
    return pd.read_excel(
        io.BytesIO(data),
        sheet_name=sheet,
        **kwargs,
    )


def file_hash(uploaded):
    return hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()


# ============================================================
# WORKBOOK LOADING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def load_workbook(data):

    wb = {}

    wb["area"] = read_excel(
        data,
        "Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    wb["cluster"] = read_excel(
        data,
        "Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    wb["ghi"] = read_excel(
        data,
        "Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    wb["forecast_config"] = read_excel(
        data,
        "Forecast Config",
        header=8,
    )

    wb["tilt"] = read_excel(
        data,
        "Config Tilt Angle",
        header=7,
    )

    wb["fixed"] = read_excel(
        data,
        "Fixed-C11",
        header=1,
    )

    wb["tracking"] = read_excel(
        data,
        "Tracking",
        header=1,
    )

    wb["backend"] = {}

    for c in CLUSTERS:
        wb["backend"][c] = read_excel(
            data,
            f"Backend Cal {c}",
        )

    return wb


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_area(df):

    df = ensure_df(df, "Area & Efficiency")

    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:
        mask = df["S.No."].isna()

        if mask.any():
            df = df.iloc[
                :np.flatnonzero(mask)[0]
            ].copy()

    required = [
        "Clusters",
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing Area columns: "
            + ", ".join(missing)
        )

    for c in required[1:]:
        df[c] = num_series(df[c])

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


def prepare_cluster(df):

    df = ensure_df(
        df,
        "Area & Efficiency cluster",
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Clusters" not in df.columns:
        raise ValueError(
            "Missing 'Clusters' column."
        )

    return (
        df.dropna(subset=["Clusters"])
        .reset_index(drop=True)
    )


def prepare_ghi(df):

    df = ensure_df(df, "Result")

    missing = [
        c for c in GHI_COLS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing GHI columns: "
            + ", ".join(missing)
        )

    for c in GHI_COLS:
        df[c] = num_series(df[c])

    return df.reset_index(drop=True)


def prepare_fixed(df):

    df = ensure_df(df, "Fixed-C11")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Actual" not in df.columns:
        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    df["Actual"] = num_series(
        df["Actual"]
    )

    return df.reset_index(drop=True)


def prepare_latitude(df):

    df = ensure_df(
        df,
        "Forecast Config",
    )

    if "Lat" not in df.columns:
        raise ValueError(
            "Latitude 'Lat' not found."
        )

    lat = pd.to_numeric(
        df["Lat"].iloc[0],
        errors="coerce",
    )

    if pd.isna(lat):
        raise ValueError(
            "Invalid latitude."
        )

    return float(lat)


def prepare_tilt(df):

    df = ensure_df(
        df,
        "Config Tilt Angle",
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Fixed" not in df.columns:
        raise ValueError(
            "Column 'Fixed' not found."
        )

    month_col = next(
        (
            c for c in df.columns
            if str(c).lower().strip() == "month"
        ),
        None,
    )

    if month_col is None:
        months = {
            "january", "february", "march",
            "april", "may", "june",
            "july", "august", "september",
            "october", "november", "december",
        }

        for c in df.columns:
            vals = (
                df[c]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            if vals.isin(months).sum() >= 2:
                month_col = c
                break

    if month_col is None:
        raise ValueError(
            "Month column not found."
        )

    result = {}

    for _, row in df.iterrows():

        month = str(
            row[month_col]
        ).strip()

        if month.lower() in {
            "january", "february", "march",
            "april", "may", "june",
            "july", "august", "september",
            "october", "november", "december",
        }:

            result[month] = safe_float(
                row["Fixed"]
            )

    if not result:
        raise ValueError(
            "No monthly tilt values found."
        )

    return result


# ============================================================
# BUILD EDITABLE INPUT
# ============================================================

def build_input(ghi, fixed):

    n = min(
        len(ghi),
        len(fixed),
    )

    if n <= 0:
        raise ValueError(
            "No GHI / Actual data available."
        )

    data = {
        c: num_array(
            ghi[c].iloc[:n]
        )
        for c in GHI_COLS
    }

    data["Actual"] = num_array(
        fixed["Actual"].iloc[:n]
    )

    return pd.DataFrame(data)


def apply_input(inp, ghi, fixed):

    required = GHI_COLS + ["Actual"]

    missing = [
        c for c in required
        if c not in inp.columns
    ]

    if missing:
        raise ValueError(
            "Missing input columns: "
            + ", ".join(missing)
        )

    n = len(inp)

    if n == 0:
        raise ValueError(
            "Input table is empty."
        )

    if n > len(ghi) or n > len(fixed):
        raise ValueError(
            "Input contains too many rows."
        )

    ghi = ghi.iloc[:n].copy()
    fixed = fixed.iloc[:n].copy()

    for c in GHI_COLS:
        ghi[c] = num_array(inp[c])

    fixed["Actual"] = num_array(
        inp["Actual"]
    )

    return (
        ghi.reset_index(drop=True),
        fixed.reset_index(drop=True),
    )


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def prepare_geometry(
    fixed,
    ghi,
    latitude,
    tilt_lookup,
):

    fixed = ensure_df(
        fixed,
        "Fixed input",
    )

    ghi = ensure_df(
        ghi,
        "GHI input",
    )

    n = min(
        len(fixed),
        len(ghi),
    )

    if n <= 0:
        raise ValueError(
            "No data available."
        )

    fixed = fixed.iloc[:n].copy()
    ghi = ghi.iloc[:n].copy()

    today = pd.Timestamp.today().normalize()

    day = today.dayofyear

    declination = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + day) / 365
            )
        )
    )

    elevation = (
        90
        - latitude
        + declination
    )

    tilt = safe_float(
        tilt_lookup.get(
            today.strftime("%B"),
            0,
        )
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt
        )
    )

    fixed["Date"] = today
    fixed["Declination Angle ∆"] = declination
    fixed["Elevation angle a"] = elevation
    fixed["Tilt Angle b"] = tilt
    fixed["a+b"] = elevation + tilt
    fixed["SIN(a+b)"] = sin_ab
    fixed["Sin(a)"] = sin_a

    # --------------------------------------------------------
    # DIRECT POA CALCULATION
    # Avoids fragile GHI*sin(a+b)-CL5 naming
    # --------------------------------------------------------

    denominator = (
        sin_a
        if abs(sin_a) > 1e-10
        else 1e-10
    )

    for i, c in enumerate(GHI_COLS):

        poa = (
            num_array(ghi[c])
            * sin_ab
            / denominator
        )

        fixed[POA_COLS[i]] = poa

    return fixed.reset_index(drop=True)


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_effective_area(
    area,
    cluster,
    error,
):

    area = ensure_df(
        area,
        "Area",
    )

    cluster = ensure_df(
        cluster,
        "Cluster",
    )

    error = safe_float(error)

    efficiency = (
        num_array(
            area["Standard PV Efficiency (%)"]
        )
        - error
    )

    total_area = num_array(
        area["Total area (m2)"]
    )

    eff_area = (
        efficiency
        * total_area
        / 100
    )

    area["Error %"] = error
    area["Net Efficiency (%)"] = efficiency
    area["Eff Area"] = eff_area

    cluster = cluster.copy()

    sums = (
        pd.DataFrame({
            "Clusters": area["Clusters"],
            "Eff Area": eff_area,
        })
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    cluster["Eff Area(m2)"] = (
        cluster["Clusters"]
        .map(sums)
        .fillna(0)
    )

    return area, cluster


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed(
    geometry,
    cluster,
):

    result = geometry.copy()

    if len(cluster) < 5:
        raise ValueError(
            "Five cluster rows are required."
        )

    for i, poa in enumerate(POA_COLS):

        if poa not in result.columns:
            raise ValueError(
                f"Missing POA '{poa}'."
            )

        area = safe_float(
            cluster.iloc[i]["Eff Area(m2)"]
        )

        result[POWER_COLS[i]] = (
            num_array(result[poa])
            * area
            / 1_000_000
        )

    result[TOTAL_POWER] = (
        result[POWER_COLS]
        .sum(axis=1)
    )

    return result


# ============================================================
# FAST FIXED ERROR OPTIMIZATION
# ============================================================
#
# Important:
# Instead of running calculate_fixed() 101 times,
# calculate the entire Error % range using arrays.
#
# This is much faster and avoids repeated DataFrame copies.
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def optimize_fixed_error(
    area,
    cluster,
    geometry,
):

    area = ensure_df(area, "Area")
    cluster = ensure_df(cluster, "Cluster")
    geometry = ensure_df(
        geometry,
        "Geometry",
    )

    actual = num_array(
        geometry["Actual"]
    )

    if len(actual) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    actual_peak = float(
        actual.max()
    )

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak must be positive."
        )

    # --------------------------------------------------------
    # Build POA matrix
    # --------------------------------------------------------

    poa_matrix = np.column_stack(
        [
            num_array(
                geometry[c]
            )
            for c in POA_COLS
        ]
    )

    # --------------------------------------------------------
    # Area calculation
    #
    # EffArea(error)
    # =
    # (Efficiency - error) * Area / 100
    # --------------------------------------------------------

    efficiency = num_array(
        area["Standard PV Efficiency (%)"]
    )

    total_area = num_array(
        area["Total area (m2)"]
    )

    cluster_names = (
        area["Clusters"]
        .astype(str)
        .to_numpy()
    )

    cluster_area = []

    for c in CLUSTERS:

        mask = (
            cluster_names
            == str(c)
        )

        base = np.sum(
            efficiency[mask]
            * total_area[mask]
            / 100
        )

        physical = np.sum(
            total_area[mask]
            / 100
        )

        cluster_area.append(
            (base, physical)
        )

    # --------------------------------------------------------
    # Forecast:
    #
    # Forecast(error)
    # =
    # Base forecast - error * loss forecast
    # --------------------------------------------------------

    base_power = np.zeros(
        len(poa_matrix)
    )

    loss_power = np.zeros(
        len(poa_matrix)
    )

    for i in range(5):

        base_eff_area = (
            cluster_area[i][0]
        )

        error_area = (
            cluster_area[i][1]
        )

        base_power += (
            poa_matrix[:, i]
            * base_eff_area
            / 1_000_000
        )

        loss_power += (
            poa_matrix[:, i]
            * error_area
            / 1_000_000
        )

    errors = np.round(
        np.arange(
            0,
            10.01,
            0.1,
        ),
        1,
    )

    forecasts = (
        base_power[:, None]
        - loss_power[:, None]
        * errors[None, :]
    )

    peaks = np.max(
        forecasts,
        axis=0,
    )

    peak_errors = np.abs(
        peaks - actual_peak
    )

    peak_error_pct = (
        peak_errors
        / actual_peak
        * 100
    )

    best_idx = int(
        np.argmin(peak_errors)
    )

    table = pd.DataFrame({
        "Error %": errors,
        "Calculated Peak": peaks,
        "Actual Peak": actual_peak,
        "Peak Error": peak_errors,
        "Peak Error %": peak_error_pct,
    })

    return (
        float(errors[best_idx]),
        table,
    )


# ============================================================
# TRACKING ARRAYS
# ============================================================

def prepare_tracking(
    backend,
    ghi,
    fixed,
    cluster,
):

    backend_c11 = ensure_df(
        backend["C11"],
        "Backend Cal C11",
    )

    if "Block No." not in backend_c11.columns:
        raise ValueError(
            "Block No. not found."
        )

    blocks = num_array(
        backend_c11["Block No."]
    )

    n = min(
        len(blocks),
        len(ghi),
        len(fixed),
    )

    if n <= 0:
        raise ValueError(
            "No Tracking data."
        )

    ghi_matrix = np.column_stack(
        [
            num_array(
                ghi[c]
            )[:n]
            for c in GHI_COLS
        ]
    )

    actual = num_array(
        fixed["Actual"]
    )[:n]

    weights = pd.to_numeric(
        cluster["Eff Area(m2)"].iloc[:5],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    if len(weights) != 5:
        raise ValueError(
            "Five tracking weights required."
        )

    return (
        blocks[:n],
        ghi_matrix,
        actual,
        weights,
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking(
    dhi,
    start,
    end,
    maximum,
    east,
    west,
    blocks,
    ghi,
    weights,
):

    if not (
        start < maximum < end
    ):
        return None

    d1 = start - 1 - maximum
    d2 = end + 1 - maximum

    if d1 == 0 or d2 == 0:
        return None

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1 * (
                blocks - maximum
            ),
        ),
        np.minimum(
            89,
            m2 * (
                blocks - maximum
            ),
        ),
    )

    panel = np.where(
        blocks < maximum,
        np.minimum(
            zenith,
            abs(east),
        ),
        np.where(
            (
                (blocks > maximum)
                & (zenith > west)
            ),
            west,
            zenith,
        ),
    )

    cos_panel = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    dhi_part = (
        ghi
        * dhi
        / 100
    )

    dni = (
        ghi - dhi_part
    ) / cos_panel[:, None]

    power = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = power.sum(
        axis=1
    )

    return (
        forecast,
        power,
        zenith,
        panel,
        dni,
    )


# ============================================================
# TRACKING OPTIMIZER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def optimize_tracking(
    blocks,
    ghi,
    actual,
    weights,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    ghi = np.asarray(
        ghi,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(ghi),
        len(actual),
    )

    blocks = blocks[:n]
    ghi = ghi[:n]
    actual = actual[:n]

    mask = actual != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual data."
        )

    actual_day = actual[mask]

    peak = actual_day.max()
    energy = actual_day.sum()

    def objective(x):

        p = np.rint(x).astype(int)

        result = calculate_tracking(
            *p,
            blocks,
            ghi,
            weights,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(prediction)
        ):
            return 1e9

        pred = prediction[mask]

        block_error = (
            np.mean(
                np.abs(
                    actual_day - pred
                )
            )
            / peak
        )

        peak_error = (
            abs(
                peak - pred.max()
            )
            / peak
        )

        energy_error = (
            abs(
                energy - pred.sum()
            )
            / energy
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        TRACKING_BOUNDS,
        strategy="best1bin",
        maxiter=40,
        popsize=10,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
        updating="immediate",
    )

    p = np.rint(
        result.x
    ).astype(int)

    return {
        "DHI": int(p[0]),
        "GHI Starting Block": int(p[1]),
        "GHI Ending Block": int(p[2]),
        "GHI Max Block": int(p[3]),
        "Tracking East Limit": int(p[4]),
        "Tracking West Limit": int(p[5]),
    }


# ============================================================
# ONE AUTOMATIC CALCULATION FUNCTION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def run_automatic_calculation(
    workbook,
    input_data,
    plant_type,
):

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    area = prepare_area(
        workbook["area"]
    )

    cluster = prepare_cluster(
        workbook["cluster"]
    )

    ghi_raw = prepare_ghi(
        workbook["ghi"]
    )

    fixed_raw = prepare_fixed(
        workbook["fixed"]
    )

    latitude = prepare_latitude(
        workbook["forecast_config"]
    )

    tilt_lookup = prepare_tilt(
        workbook["tilt"]
    )

    # --------------------------------------------------------
    # APPLY USER INPUT
    # --------------------------------------------------------

    ghi, fixed = apply_input(
        input_data,
        ghi_raw,
        fixed_raw,
    )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    geometry = prepare_geometry(
        fixed,
        ghi,
        latitude,
        tilt_lookup,
    )

    # --------------------------------------------------------
    # AUTOMATIC FIXED ERROR
    # --------------------------------------------------------

    best_error, error_table = (
        optimize_fixed_error(
            area,
            cluster,
            geometry,
        )
    )

    # --------------------------------------------------------
    # APPLY BEST ERROR
    # --------------------------------------------------------

    final_area, final_cluster = (
        calculate_effective_area(
            area,
            cluster,
            best_error,
        )
    )

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    fixed_result = calculate_fixed(
        geometry,
        final_cluster,
    )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    tracking_params = None

    if plant_type == "Tracking":

        (
            blocks,
            ghi_matrix,
            actual,
            weights,
        ) = prepare_tracking(
            workbook["backend"],
            ghi,
            geometry,
            final_cluster,
        )

        tracking_params = optimize_tracking(
            tuple(blocks.tolist()),
            tuple(
                tuple(x)
                for x in ghi_matrix
            ),
            tuple(actual.tolist()),
            tuple(weights.tolist()),
        )

    return {
        "area": area,
        "cluster": cluster,
        "final_area": final_area,
        "final_cluster": final_cluster,
        "ghi": ghi,
        "fixed": geometry,
        "fixed_result": fixed_result,
        "best_error": best_error,
        "error_table": error_table,
        "tracking_params": tracking_params,
    }


# ============================================================
# FINAL EDITABLE FORECAST
# ============================================================

def calculate_final_forecast(
    data,
    workbook,
    plant_type,
    error,
    tracking_params=None,
):

    # --------------------------------------------------------
    # ERROR %
    # --------------------------------------------------------

    _, cluster = calculate_effective_area(
        data["area"],
        data["cluster"],
        error,
    )

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    fixed_result = calculate_fixed(
        data["fixed"],
        cluster,
    )

    actual = num_array(
        data["fixed"]["Actual"]
    )

    if plant_type == "Fixed":

        forecast = num_array(
            fixed_result[TOTAL_POWER]
        )

        return (
            actual,
            forecast,
            fixed_result,
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    if tracking_params is None:
        raise ValueError(
            "Tracking parameters unavailable."
        )

    (
        blocks,
        ghi_matrix,
        actual_tracking,
        weights,
    ) = prepare_tracking(
        workbook["backend"],
        data["ghi"],
        data["fixed"],
        cluster,
    )

    p = tracking_params

    result = calculate_tracking(
        int(p["DHI"]),
        int(p["GHI Starting Block"]),
        int(p["GHI Ending Block"]),
        int(p["GHI Max Block"]),
        int(p["Tracking East Limit"]),
        int(p["Tracking West Limit"]),
        blocks,
        ghi_matrix,
        weights,
    )

    if result is None:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    return (
        actual_tracking,
        result[0],
        fixed_result,
    )


# ============================================================
# GRAPH
# ============================================================

def make_graph(
    actual,
    forecast,
    title,
):

    n = min(
        len(actual),
        len(forecast),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual[:n],
            mode="lines",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast[:n],
            mode="lines",
            name="Forecast",
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=30,
            r=30,
            t=55,
            b=30,
        ),
        xaxis_title="Block",
        yaxis_title="Power",
    )

    return fig
