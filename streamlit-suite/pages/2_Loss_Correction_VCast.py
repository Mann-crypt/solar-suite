import io
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution


CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]
GHI_COLS = [f"GHI {c}" for c in CLUSTERS]
REQUIRED_VCAST_SHEETS = [
    "Fixed-C11", "Area & Efficiency", "Forecast Config",
    "Config Tilt Angle", "Result"
]

def file_id(uploaded_file):
    data = uploaded_file.getvalue()

    return (
        uploaded_file.name,
        len(data),
        hashlib.md5(data).hexdigest(),
    )
def numeric_array(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )
def create_time_blocks(count=96):
    start = datetime.strptime(
        "00:00",
        "%H:%M",
    )

    return [
        (
            f"{(start + timedelta(minutes=15 * i)).strftime('%H:%M')}"
            f" - "
            f"{(start + timedelta(minutes=15 * (i + 1))).strftime('%H:%M')}"
        )
        for i in range(count)
    ]
def calculate_metrics(actual, forecast):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "Actual power contains no valid non-zero values."
        )

    actual_day = actual[mask]
    forecast_day = forecast[mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    if actual_energy == 0:
        raise ValueError(
            "Actual energy must be greater than zero."
        )

    block_error = (
        np.mean(
            np.abs(
                actual_day
                -
                forecast_day
            )
        )
        /
        actual_peak
    )

    peak_error = (
        abs(
            actual_peak
            -
            forecast_day.max()
        )
        /
        actual_peak
    )

    energy_error = (
        abs(
            actual_energy
            -
            forecast_day.sum()
        )
        /
        actual_energy
    )

    score = (
        0.80 * block_error
        +
        0.10 * peak_error
        +
        0.10 * energy_error
    )

    return {
        "mask": mask,
        "actual_peak": actual_peak,
        "forecast_peak": forecast_day.max(),
        "block_error": block_error,
        "peak_error": peak_error,
        "energy_error": energy_error,
        "score": score,
    }
def read_vcast_workbook(file_bytes):
    """
    Read the VCast workbook once.

    This intentionally keeps the VCast sheet structure separate
    from the normal Fixed / Cluster workbook format.
    """

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    sheet_names = excel.sheet_names

    missing = [
        sheet
        for sheet in REQUIRED_VCAST_SHEETS
        if sheet not in sheet_names
    ]

    if missing:
        raise ValueError(
            "Missing VCast sheet(s): "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    area_raw = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
    )

    area_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area_df.columns = (
        area_df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    if "Standard PV Efficiency (%)" not in area_df.columns:
        raise ValueError(
            "Column 'Standard PV Efficiency (%)' "
            "was not found in Area & Efficiency."
        )

    # --------------------------------------------------------
    # VCAST EFFECTIVE AREAS
    #
    # Fixed     : P3:P7
    # Tracking  : P29:P33
    # --------------------------------------------------------

    fixed_weights = (
        pd.to_numeric(
            area_raw.iloc[
                2:7,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_weights = (
        pd.to_numeric(
            area_raw.iloc[
                28:33,
                15,
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(fixed_weights) != 5:
        raise ValueError(
            "Could not read 5 Fixed effective-area values "
            "from Area & Efficiency column P."
        )

    if len(tracking_weights) != 5:
        raise ValueError(
            "Could not read 5 Tracking effective-area values "
            "from Area & Efficiency column P."
        )

    standard_efficiency = numeric_array(
        area_df[
            "Standard PV Efficiency (%)"
        ]
    )[:5]

    if len(standard_efficiency) != 5:
        raise ValueError(
            "Could not read 5 Standard PV Efficiency values."
        )

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    if "Lat" not in config.columns:
        raise ValueError(
            "Column 'Lat' was not found in Forecast Config."
        )

    lat = float(
        pd.to_numeric(
            config.loc[
                0,
                "Lat",
            ],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # CONFIG TILT ANGLE
    # --------------------------------------------------------

    tilt_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    tilt_df.columns = (
        tilt_df.columns
        .astype(str)
        .str.strip()
    )

    tilt_df = tilt_df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if "Fixed" not in tilt_df.columns:
        raise ValueError(
            "Column 'Fixed' was not found in Config Tilt Angle."
        )

    tilt_df["Month_Num"] = pd.to_numeric(
        tilt_df["Month_Num"],
        errors="coerce",
    )

    tilt_df["Fixed"] = pd.to_numeric(
        tilt_df["Fixed"],
        errors="coerce",
    )

    tilt_lookup = (
        tilt_df
        .dropna(
            subset=[
                "Month_Num",
                "Fixed",
            ]
        )
        .set_index(
            "Month_Num"
        )["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # RESULT / GHI
    # --------------------------------------------------------

    ghi_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=range(6),
    )

    ghi_df.columns = [
        "Block",
        *GHI_COLS,
    ]

    ghi_df["Block"] = pd.to_numeric(
        ghi_df["Block"],
        errors="coerce",
    )

    ghi_df = ghi_df[
        ghi_df["Block"].notna()
    ].copy()

    for col in GHI_COLS:
        ghi_df[col] = pd.to_numeric(
            ghi_df[col],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # FIXED-C11
    #
    # Stop at first blank Date.
    # --------------------------------------------------------

    fixed_df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    fixed_df.columns = (
        fixed_df.columns
        .astype(str)
        .str.strip()
    )

    if "Date" not in fixed_df.columns:
        raise ValueError(
            "Column 'Date' was not found in Fixed-C11."
        )

    if "Actual" not in fixed_df.columns:
        raise ValueError(
            "Column 'Actual' was not found in Fixed-C11."
        )

    date_valid = fixed_df["Date"].notna()

    if not date_valid.any():
        raise ValueError(
            "No valid Date rows found in Fixed-C11."
        )

    first_blank = np.where(
        ~date_valid.to_numpy()
    )[0]

    if len(first_blank) > 0:
        fixed_df = fixed_df.iloc[
            :first_blank[0]
        ].copy()
    else:
        fixed_df = fixed_df.loc[
            date_valid
        ].copy()

    fixed_df.reset_index(
        drop=True,
        inplace=True,
    )

    fixed_df["Date"] = pd.to_datetime(
        fixed_df["Date"],
        errors="coerce",
    )

    if fixed_df["Date"].isna().any():
        raise ValueError(
            "Invalid dates found in Fixed-C11."
        )

    # --------------------------------------------------------
    # ALIGN DATA
    # --------------------------------------------------------

    n = min(
        len(fixed_df),
        len(ghi_df),
    )

    if n == 0:
        raise ValueError(
            "No aligned VCast rows are available."
        )

    fixed_df = fixed_df.iloc[
        :n
    ].copy()

    ghi_df = ghi_df.iloc[
        :n
    ].copy()

    actual = numeric_array(
        fixed_df["Actual"]
    )[:n]

    blocks = ghi_df[
        "Block"
    ].to_numpy(
        dtype=float
    )[:n]

    ghi_matrix = ghi_df[
        GHI_COLS
    ].to_numpy(
        dtype=float
    )[:n]

    dates = fixed_df[
        "Date"
    ]

    # --------------------------------------------------------
    # SOLAR GEOMETRY
    # --------------------------------------------------------

    first_date = pd.Timestamp(
        "2025-01-01"
    )

    day_offset = (
        dates
        -
        first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    declination = (
        23.45
        *
        np.sin(
            np.radians(
                360
                *
                (
                    284
                    +
                    day_offset
                    +
                    1
                )
                /
                365
            )
        )
    )

    elevation = (
        90
        -
        lat
        +
        declination
    )

    months = (
        dates
        .dt
        .month
        .to_numpy()
    )

    tilt = np.array(
        [
            tilt_lookup.get(
                float(month),
                0,
            )
            for month in months
        ]
    )

    sin_a = np.sin(
        np.radians(
            elevation
        )
    )

    sin_ab = np.sin(
        np.radians(
            elevation
            +
            tilt
        )
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a,
    )

    fixed_geometry_factor = (
        sin_ab
        /
        sin_a_safe
    )

    fixed_poa = (
        ghi_matrix
        *
        fixed_geometry_factor[:, None]
    )

    # --------------------------------------------------------
    # TRACKING SHEET
    # --------------------------------------------------------

    tracking_available = (
        "Tracking" in sheet_names
    )

    tracking_df = None

    if tracking_available:
        tracking_df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Tracking",
            header=1,
        )

        tracking_df.columns = (
            tracking_df.columns
            .astype(str)
            .str.strip()
        )

        tracking_df = tracking_df.iloc[
            :n
        ].copy()

        tracking_df.reset_index(
            drop=True,
            inplace=True,
        )

    return {
        "sheet_names": sheet_names,
        "area_df": area_df,
        "fixed_weights": fixed_weights,
        "tracking_weights": tracking_weights,
        "standard_efficiency": standard_efficiency,
        "lat": lat,
        "tilt_lookup": tilt_lookup,
        "fixed_df": fixed_df,
        "ghi_df": ghi_df,
        "actual": actual,
        "blocks": blocks,
        "ghi_matrix": ghi_matrix,
        "fixed_poa": fixed_poa,
        "tracking_available": tracking_available,
        "tracking_df": tracking_df,
    }
def optimize_fixed(
    standard_efficiency_tuple,
    fixed_weights_tuple,
    fixed_poa_tuple,
    actual_tuple,
):
    std_eff = np.asarray(
        standard_efficiency_tuple,
        dtype=float,
    )

    fixed_weights = np.asarray(
        fixed_weights_tuple,
        dtype=float,
    )

    fixed_poa = np.asarray(
        fixed_poa_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No valid actual values available for Fixed optimization."
        )

    actual_day = actual[mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    results = []

    max_loss = float(
        np.min(
            std_eff
        )
    )

    for loss in np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    ):
        net_efficiency = np.maximum(
            std_eff - loss,
            0,
        )

        efficiency_factor = np.divide(
            net_efficiency,
            std_eff,
            out=np.zeros_like(
                net_efficiency
            ),
            where=(
                std_eff != 0
            ),
        )

        final_weights = (
            fixed_weights
            *
            efficiency_factor
        )

        power_matrix = (
            fixed_poa
            *
            final_weights[None, :]
            /
            1_000_000
        )

        forecast = (
            power_matrix.sum(
                axis=1
            )
        )

        forecast_day = forecast[
            mask
        ]

        predicted_peak = (
            forecast_day.max()
        )

        peak_error = abs(
            actual_peak
            -
            predicted_peak
        )

        peak_error_percent = (
            peak_error
            /
            actual_peak
            *
            100
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    -
                    forecast_day
                )
            )
            /
            actual_peak
        )

        predicted_energy = (
            forecast_day.sum()
        )

        energy_error = (
            abs(
                actual_energy
                -
                predicted_energy
            )
            /
            actual_energy
        )

        overall_score = (
            0.80 * block_error
            +
            0.10 * (
                peak_error
                /
                actual_peak
            )
            +
            0.10 * energy_error
        )

        results.append(
            {
                "Error %": loss,
                "Actual Peak": actual_peak,
                "Predicted Peak": predicted_peak,
                "Peak Error": peak_error,
                "Peak Error (%)": peak_error_percent,
                "Block Error": block_error,
                "Energy Error": energy_error,
                "Overall Score": overall_score,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise ValueError(
            "Fixed optimization produced no results."
        )

    # IMPORTANT:
    # VCast validated logic selects minimum Peak Error.
    best_row = results_df.loc[
        results_df[
            "Peak Error"
        ].idxmin()
    ]

    best_loss = float(
        best_row[
            "Error %"
        ]
    )

    final_net_efficiency = np.maximum(
        std_eff - best_loss,
        0,
    )

    final_efficiency_factor = np.divide(
        final_net_efficiency,
        std_eff,
        out=np.zeros_like(
            std_eff
        ),
        where=(
            std_eff != 0
        ),
    )

    final_weights = (
        fixed_weights
        *
        final_efficiency_factor
    )

    final_power_matrix = (
        fixed_poa
        *
        final_weights[None, :]
        /
        1_000_000
    )

    final_forecast = (
        final_power_matrix.sum(
            axis=1
        )
    )

    return (
        best_loss,
        final_forecast,
        final_power_matrix,
        final_net_efficiency,
        results_df,
    )
def calculate_tracking(
    blocks,
    ghi_matrix,
    tracking_weights,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):
    if not (
        start_block
        <
        max_block
        <
        end_block
    ):
        return None

    denominator_1 = (
        start_block
        -
        1
        -
        max_block
    )

    denominator_2 = (
        end_block
        +
        1
        -
        max_block
    )

    if (
        denominator_1 == 0
        or
        denominator_2 == 0
    ):
        return None

    m1 = (
        90
        /
        denominator_1
    )

    m2 = (
        90
        /
        denominator_2
    )

    zenith = np.where(
        blocks <= max_block,
        np.minimum(
            89,
            m1
            *
            (
                blocks
                -
                max_block
            ),
        ),
        np.minimum(
            89,
            m2
            *
            (
                blocks
                -
                max_block
            ),
        ),
    )

    panel = np.where(
        blocks < max_block,
        np.where(
            zenith
            <
            abs(
                east_limit
            ),
            zenith,
            abs(
                east_limit
            ),
        ),
        np.where(
            (
                (blocks > max_block)
                &
                (zenith > west_limit)
            ),
            west_limit,
            zenith,
        ),
    )

    cos_alpha = np.cos(
        np.radians(
            panel
        )
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        *
        DHI
        /
        100
    )

    dni = (
        ghi_matrix
        -
        dhi
    ) / cos_alpha[:, None]

    tracking_power_matrix = (
        dni
        *
        tracking_weights[None, :]
        /
        1_000_000
    )

    tracking_forecast = (
        tracking_power_matrix.sum(
            axis=1
        )
    )

    return (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
    )
def optimize_tracking(
    blocks_tuple,
    ghi_matrix_tuple,
    tracking_weights_tuple,
    actual_tuple,
):
    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi_matrix = np.asarray(
        ghi_matrix_tuple,
        dtype=float,
    )

    tracking_weights = np.asarray(
        tracking_weights_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No valid actual values available for Tracking optimization."
        )

    actual_day = actual[
        mask
    ]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    def objective(x):
        DHI = int(
            round(
                x[0]
            )
        )

        start_block = int(
            round(
                x[1]
            )
        )

        end_block = int(
            round(
                x[2]
            )
        )

        max_block = int(
            round(
                x[3]
            )
        )

        east_limit = int(
            round(
                x[4]
            )
        )

        west_limit = int(
            round(
                x[5]
            )
        )

        result = calculate_tracking(
            blocks,
            ghi_matrix,
            tracking_weights,
            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
        )

        if result is None:
            return 1e9

        prediction = result[0]

        if not np.all(
            np.isfinite(
                prediction
            )
        ):
            return 1e9

        prediction_day = prediction[
            mask
        ]

        if len(prediction_day) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    -
                    prediction_day
                )
            )
            /
            actual_peak
        )

        peak_error = (
            abs(
                actual_peak
                -
                prediction_day.max()
            )
            /
            actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                -
                prediction_day.sum()
            )
            /
            actual_energy
        )

        return (
            0.80 * block_error
            +
            0.10 * peak_error
            +
            0.10 * energy_error
        )

    bounds = [
        (0, 10),      # DHI
        (10, 30),     # GHI Start
        (65, 80),     # GHI End
        (47, 53),     # GHI Max
        (10, 70),     # East Limit
        (10, 70),     # West Limit
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
        result.x
    ).astype(int)

    rounded_score = objective(
        best
    )

    return (
        tuple(
            best.tolist()
        ),
        float(
            result.fun
        ),
        float(
            rounded_score
        ),
    )

# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="VCast Loss Correction", page_icon="☀️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stFileUploaderDropzone"] {border-radius: 12px;}
[data-testid="stDataEditor"] {border: 1px solid #e5e7eb; border-radius: 12px;}
.stButton > button {border-radius: 10px; font-weight: 700; min-height: 42px;}
.section {font-size: 20px; font-weight: 700; margin: 18px 0 10px;}
.card {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;}
.small {color:#6b7280; font-size:13px;}
</style>
""", unsafe_allow_html=True)

st.markdown("# ☀️ VCast Loss Correction")
st.caption("Automatic Fixed / Tracking forecast correction with editable inputs and parameters")

if "vc_file_id" not in st.session_state:
    st.session_state.vc_file_id = None
    st.session_state.vc_run = False
    st.session_state.vc_result = None
    st.session_state.vc_editor_key = 0

uploaded = st.file_uploader("VCast Excel Workbook", type=["xlsx", "xls"], width="stretch")
if uploaded is None:
    st.info("Upload the VCast workbook to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
current_id = file_id(uploaded)
if current_id != st.session_state.vc_file_id:
    st.session_state.vc_file_id = current_id
    st.session_state.vc_run = False
    st.session_state.vc_result = None
    st.session_state.vc_editor_key += 1

try:
    workbook = read_vcast_workbook(file_bytes)
except Exception as e:
    st.error(f"Unable to read workbook: {e}")
    st.stop()

# ------------------------------------------------------------
# INPUT + RUN FORM
# ------------------------------------------------------------

st.markdown('<div class="section">📥 Input Data</div>', unsafe_allow_html=True)

base_input = pd.DataFrame({"Date": workbook["fixed_df"]["Date"], "Actual": workbook["actual"]})
for col in GHI_COLS:
    base_input[col] = workbook["ghi_df"][col].to_numpy()

with st.form(f"vc_input_form_{st.session_state.vc_editor_key}", clear_on_submit=False):
    edited_df = st.data_editor(
        base_input,
        hide_index=True,
        num_rows="fixed",
        height=360,
        width="stretch",
        key=f"vc_editor_{st.session_state.vc_editor_key}",
        disabled=["Date"],
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD-MM-YYYY"),
            **{c: st.column_config.NumberColumn(c, format="%.4f") for c in ["Actual", *GHI_COLS]},
        },
    )

    st.markdown('<div class="section">🌞 Plant Type</div>', unsafe_allow_html=True)
    plant_type = st.segmented_control(
        "Plant Type",
        ["Fixed", "Tracking"],
        default="Fixed",
        selection_mode="single",
        width="stretch",
        label_visibility="collapsed",
    ) or "Fixed"

    run_clicked = st.form_submit_button(
        "⚡ Run Automatic Calculation",
        type="primary",
        use_container_width=True,
    )

# ------------------------------------------------------------
# CLEAN EDITED INPUT
# ------------------------------------------------------------

edited_df = edited_df.copy()
edited_df["Actual"] = pd.to_numeric(edited_df["Actual"], errors="coerce").fillna(0)
for col in GHI_COLS:
    edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce").fillna(0)

n = min(len(edited_df), len(workbook["fixed_poa"]))
edited_df = edited_df.iloc[:n].copy()
actual = edited_df["Actual"].to_numpy(dtype=float)
ghi_matrix = edited_df[GHI_COLS].to_numpy(dtype=float)
blocks = workbook["blocks"][:n]

original_ghi = workbook["ghi_matrix"][:n]
geometry_factor = np.divide(
    workbook["fixed_poa"][:n], original_ghi,
    out=np.zeros_like(original_ghi),
    where=np.abs(original_ghi) > 1e-12,
)
fixed_poa = ghi_matrix * geometry_factor

# ------------------------------------------------------------
# AUTOMATIC OPTIMIZATION: ONLY ON FORM SUBMIT
# ------------------------------------------------------------

if run_clicked:
    try:
        with st.spinner("Running automatic calculation..."):
            fixed_result = optimize_fixed(
                tuple(workbook["standard_efficiency"]),
                tuple(workbook["fixed_weights"]),
                tuple(fixed_poa.tolist()),
                tuple(actual.tolist()),
            )

            tracking_result = None
            if plant_type == "Tracking":
                if not workbook["tracking_available"]:
                    raise ValueError("Tracking sheet was not found in this VCast workbook.")
                tracking_result = optimize_tracking(
                    tuple(blocks.tolist()),
                    tuple(ghi_matrix.tolist()),
                    tuple(workbook["tracking_weights"].tolist()),
                    tuple(actual.tolist()),
                )

        st.session_state.vc_result = {
            "plant_type": plant_type,
            "actual": actual,
            "ghi_matrix": ghi_matrix,
            "fixed_poa": fixed_poa,
            "blocks": blocks,
            "fixed_best_loss": float(fixed_result[0]),
            "tracking_best": tracking_result[0] if tracking_result else None,
        }
        st.session_state.vc_run = True
        st.success("Automatic calculation completed.")
    except Exception as e:
        st.error(f"Calculation failed: {e}")
        st.stop()

if not st.session_state.vc_run or st.session_state.vc_result is None:
    st.info("Edit the input data, choose Fixed or Tracking, then click Run Automatic Calculation.")
    st.stop()

# ------------------------------------------------------------
# RESULT STATE
# ------------------------------------------------------------

result = st.session_state.vc_result
actual = result["actual"]
ghi_matrix = result["ghi_matrix"]
fixed_poa = result["fixed_poa"]
blocks = result["blocks"]

# Keep the current selector visually consistent after reruns.
current_plant = plant_type

st.markdown('<div class="section">⚙️ Parameters</div>', unsafe_allow_html=True)

if current_plant == "Fixed":
    error_value = st.number_input(
        "Efficiency Error %",
        min_value=0.0,
        max_value=float(np.min(workbook["standard_efficiency"])),
        value=float(result["fixed_best_loss"]),
        step=0.1,
        format="%.1f",
        key="vc_fixed_error",
    )

    std_eff = np.asarray(workbook["standard_efficiency"], dtype=float)
    fixed_weights = np.asarray(workbook["fixed_weights"], dtype=float)
    net_eff = np.maximum(std_eff - error_value, 0)
    factor = np.divide(net_eff, std_eff, out=np.zeros_like(std_eff), where=std_eff != 0)
    final_weights = fixed_weights * factor
    fixed_power_matrix = fixed_poa * final_weights[None, :] / 1_000_000
    forecast = fixed_power_matrix.sum(axis=1)
    title = "Fixed Plant | Actual vs Forecast"
    parameter_text = f"Automatic loss: {result['fixed_best_loss']:.1f}%  •  Current loss: {error_value:.1f}%"

else:
    if not workbook["tracking_available"]:
        st.error("Tracking sheet was not found in this VCast workbook.")
        st.stop()

    best = result["tracking_best"]
    d0, s0, e0, m0, east0, west0 = [int(x) for x in best]
    p1, p2, p3 = st.columns(3)
    with p1:
        dhi = st.number_input("DHI (%)", 0, 100, d0, 1)
        start_block = st.number_input("GHI Starting Block", 0, 95, s0, 1)
    with p2:
        end_block = st.number_input("GHI Ending Block", 1, 96, e0, 1)
        max_block = st.number_input("GHI Max Block", 0, 95, m0, 1)
    with p3:
        east_limit = st.number_input("Tracking East Limit", 0, 90, east0, 1)
        west_limit = st.number_input("Tracking West Limit", 0, 90, west0, 1)

    tracking_weights = np.asarray(workbook["tracking_weights"], dtype=float)
    tracking_output = calculate_tracking(
        blocks, ghi_matrix, tracking_weights,
        int(dhi), int(start_block), int(end_block), int(max_block),
        int(east_limit), int(west_limit),
    )
    if tracking_output is None:
        st.error("Invalid tracking parameters. Keep GHI Starting < GHI Max < GHI Ending.")
        st.stop()
    forecast = tracking_output[0]
    title = "Tracking Plant | Actual vs Forecast"
    parameter_text = f"DHI: {int(dhi)}%  •  GHI Max: {int(max_block)}  •  East: {int(east_limit)}°  •  West: {int(west_limit)}°"

# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

metrics = calculate_metrics(actual, forecast)

st.markdown('<div class="section">📊 Results</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Actual Peak", f"{metrics['actual_peak']:.3f}")
with c2:
    st.metric("Forecast Peak", f"{metrics['forecast_peak']:.3f}")
with c3:
    st.metric("Peak Error", f"{metrics['peak_error'] * 100:.2f}%")

st.caption(parameter_text)

fig = go.Figure()
fig.add_trace(go.Scatter(x=blocks, y=actual, mode="lines", name="Actual", line=dict(width=2.5)))
fig.add_trace(go.Scatter(x=blocks, y=forecast, mode="lines", name="Forecast", line=dict(width=2.5)))
fig.update_layout(
    height=470,
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=55, b=20),
    xaxis_title="Block",
    yaxis_title="Power (MW)",
    legend=dict(orientation="h", y=1.02, x=0),
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
