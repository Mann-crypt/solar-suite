import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Loss Correction",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .sub-title {
        color: #6b7280;
        font-size: 15px;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Loss Correction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Forecast correction and efficiency-loss optimization platform'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def file_hash(uploaded_file):
    """
    Generate a unique hash for the uploaded workbook.

    This prevents Streamlit from confusing two different files
    having the same filename.
    """
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


@st.cache_data(show_spinner=False)
def get_excel_sheet(file_bytes, sheet_name, **kwargs):
    """
    Cached Excel reader.

    Excel sheets are loaded only when the workbook actually changes.
    """
    return pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
        **kwargs
    )


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes):
    """
    Cached workbook sheet-name reader.
    """
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


def clean_until_empty(df, column):
    """
    Remove rows after the first empty value in a specified column.
    """
    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[df[column].isna()].index

    if len(null_indices) > 0:
        first_null = null_indices[0]
        df = df.loc[:first_null - 1]

    return df.reset_index(drop=True)


def prepare_area_efficiency(file_bytes, is_cluster):
    """
    Read Area & Efficiency sheet.
    """

    if is_cluster:
        df = get_excel_sheet(
            file_bytes,
            "Area & Efficiency",
            header=1,
            usecols=range(8)
        )
    else:
        df = get_excel_sheet(
            file_bytes,
            "Area & Efficiency",
            header=1
        )

    df.columns = df.columns.astype(str).str.strip()

    df = clean_until_empty(df, "Module Type")

    numeric_columns = [
        "Standard PV Efficiency (%)",
        "Total area(m2)"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    return df


def get_latitude(file_bytes):
    """
    Read latitude from Forecast Config.
    """

    df_st = get_excel_sheet(
        file_bytes,
        "Forecast Config",
        header=8
    )

    return float(df_st.loc[0, "Lat"])


def prepare_fixed_ghi_data(
    file_bytes,
    edited_ghi,
    edited_actual,
    is_cluster
):
    """
    Prepare the 96-block forecast/actual dataframe.
    """

    sheet = "Fixed-CL1" if is_cluster else "Fixed"

    df_fix = get_excel_sheet(
        file_bytes,
        sheet,
        header=1
    ).copy()

    df_fix.columns = df_fix.columns.astype(str).str.strip()

    df_fix = clean_until_empty(df_fix, "Date")

    df_fix = df_fix.iloc[:96].copy()

    if is_cluster:

        ghi_columns = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI"
        ]

    else:

        ghi_columns = [
            "GHI_Forecast"
        ]

    for col in ghi_columns:
        df_fix[col] = pd.to_numeric(
            edited_ghi[col],
            errors="coerce"
        ).fillna(0).to_numpy()

    df_fix["Actual"] = pd.to_numeric(
        edited_actual["Actual"],
        errors="coerce"
    ).fillna(0).to_numpy()

    return df_fix, ghi_columns


def calculate_efficiency_loss(
    df,
    df_fix,
    ghi_columns,
    is_cluster,
    cluster_weights=None
):
    """
    Find the best efficiency loss.

    We intentionally do NOT expose or calculate MAE, MAPE,
    Peak Error, R2, or block error.

    The internal optimization simply finds the efficiency loss
    that produces the closest overall forecast curve to Actual.
    """

    standard_efficiency = df[
        "Standard PV Efficiency (%)"
    ].to_numpy(dtype=np.float64)

    area = df[
        "Total area(m2)"
    ].to_numpy(dtype=np.float64)

    actual = df_fix[
        "Actual"
    ].to_numpy(dtype=np.float64)

    if actual.size == 0:
        return 0.0

    # --------------------------------------------------------
    # Precalculate irradiation component
    # --------------------------------------------------------

    poa_data = []

    for col in ghi_columns:

        ghi = df_fix[col].to_numpy(
            dtype=np.float64
        )

        poa_data.append(ghi)

    poa_data = np.asarray(poa_data)

    # --------------------------------------------------------
    # Loss candidates
    # --------------------------------------------------------

    max_loss = max(
        0,
        float(np.min(standard_efficiency))
    )

    loss_candidates = np.arange(
        0,
        max_loss + 0.001,
        0.1
    )

    best_loss = 0.0
    best_score = np.inf

    # --------------------------------------------------------
    # Optimize efficiency loss
    # --------------------------------------------------------

    for loss in loss_candidates:

        net_efficiency = (
            standard_efficiency - loss
        )

        net_efficiency = np.maximum(
            net_efficiency,
            0
        )

        if is_cluster:

            if cluster_weights is None:
                continue

            effective_area = np.array([
                np.sum(
                    area
                    * net_efficiency
                    / 100
                    * cluster_weights[i]
                )
                for i in range(5)
            ])

        else:

            effective_area = np.array([
                np.sum(
                    area
                    * net_efficiency
                    / 100
                )
            ])

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        forecast = np.zeros(
            len(df_fix),
            dtype=np.float64
        )

        for i in range(len(ghi_columns)):

            ghi = poa_data[i]

            if is_cluster:
                power = (
                    ghi
                    * effective_area[i]
                )
            else:
                power = (
                    ghi
                    * effective_area[0]
                )

            forecast += power / 1_000_000

        # ----------------------------------------------------
        # Internal fitting score
        # ----------------------------------------------------

        valid = np.isfinite(
            actual
        ) & np.isfinite(
            forecast
        )

        if not valid.any():
            continue

        actual_valid = actual[valid]
        forecast_valid = forecast[valid]

        scale = max(
            np.max(np.abs(actual_valid)),
            1e-6
        )

        score = np.mean(
            np.abs(
                actual_valid - forecast_valid
            )
        ) / scale

        if score < best_score:

            best_score = score
            best_loss = float(loss)

    return best_loss


def calculate_fixed_forecast(
    df,
    df_fix,
    best_loss,
    latitude,
    month_lookup=None
):
    """
    Fast fixed-plant final calculation.
    """

    df = df.copy()

    df["Efficiency Losses(%)"] = best_loss

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Efficiency Losses(%)"]
    )

    df["Net Efficiency (%)"] = np.maximum(
        df["Net Efficiency (%)"],
        0
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
    ) / 100

    # --------------------------------------------------------
    # Solar geometry
    # --------------------------------------------------------

    date_value = pd.Timestamp.today().normalize()

    df_fix = df_fix.copy()

    df_fix["Date"] = date_value

    first_date = date_value.replace(
        month=1,
        day=1
    )

    day_number = (
        df_fix["Date"] - first_date
    ).dt.days + 1

    declination = 23.45 * np.sin(
        np.radians(
            360
            * (284 + day_number)
            / 365
        )
    )

    df_fix["Declination Angle ∆"] = declination

    df_fix["Elevation angle a"] = (
        90 - latitude + declination
    )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    if month_lookup is not None:

        month_name = date_value.strftime("%B")

        tilt = month_lookup.get(
            month_name,
            0
        )

    else:

        tilt = 0

    df_fix["Tilt Angle b"] = tilt

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    sin_a = np.sin(
        np.radians(
            df_fix["Elevation angle a"]
        )
    )

    sin_ab = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    sin_a = np.clip(
        sin_a,
        1e-6,
        None
    )

    ghi = df_fix[
        "GHI_Forecast"
    ].to_numpy(
        dtype=np.float64
    )

    poa = (
        ghi * sin_ab
    ) / sin_a

    effective_area = df["Eff Area"].sum()

    forecast = (
        poa
        * effective_area
    ) / 1_000_000

    return (
        df,
        df_fix,
        forecast
    )


def calculate_tracking_forecast(
    df,
    df_fix,
    best_loss,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
    backend_blocks
):
    """
    Fast vectorized tracking calculation.

    This function is called whenever the user changes one
    of the optimized parameters.

    It DOES NOT run Differential Evolution.
    """

    df = df.copy()

    df["Efficiency Losses(%)"] = best_loss

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - df["Efficiency Losses(%)"]
    )

    df["Net Efficiency (%)"] = np.maximum(
        df["Net Efficiency (%)"],
        0
    )

    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
    ) / 100

    effective_area = df[
        "Eff Area"
    ].sum()

    blocks = np.asarray(
        backend_blocks,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    if not (
        start_block < max_block < end_block
    ):
        return (
            df,
            np.zeros(len(df_fix))
        )

    # --------------------------------------------------------
    # Tracking geometry
    # --------------------------------------------------------

    denominator_1 = (
        start_block
        - 1
        - max_block
    )

    denominator_2 = (
        end_block
        + 1
        - max_block
    )

    if denominator_1 == 0 or denominator_2 == 0:

        return (
            df,
            np.zeros(len(df_fix))
        )

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1 * (
                blocks - max_block
            )
        ),

        np.minimum(
            89,
            m2 * (
                blocks - max_block
            )
        )
    )

    panel = np.where(
        blocks < max_block,

        np.minimum(
            zenith,
            abs(east_limit)
        ),

        np.where(
            (
                (blocks > max_block)
                &
                (zenith > west_limit)
            ),
            west_limit,
            zenith
        )
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    ghi = df_fix[
        "GHI_Forecast"
    ].to_numpy(
        dtype=np.float64
    )

    dhi = (
        ghi
        * DHI
        / 100
    )

    dni = (
        ghi - dhi
    ) / cos_alpha

    forecast = (
        dni
        * effective_area
    ) / 1_000_000

    return (
        df,
        forecast
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style='text-align:center;
        background:linear-gradient(90deg,#00c6ff,#0072ff);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-size:38px;
        font-weight:800;'>
        ⚡ Solar Suite
        </h1>

        <p style='text-align:center;color:gray'>
        Loss Correction
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div style='text-align:center;color:gray;font-size:13px'>
        Developed and Maintained by:<br>
        <b>Manjot Singh</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx"],
    key="loss_excel"
)

if uploaded_file is None:

    st.info(
        "Upload the solar workbook to start Loss Correction."
    )

    st.stop()


# ============================================================
# FILE IDENTIFICATION
# ============================================================

file_bytes = uploaded_file.getvalue()

current_hash = file_hash(
    uploaded_file
)

# Reset calculation state when a genuinely new workbook arrives.

if st.session_state.get(
    "loss_file_hash"
) != current_hash:

    keys_to_remove = [
        "loss_file_hash",
        "loss_calculated",
        "optimization_result",
        "loss_params",
        "loss_editor"
    ]

    for key in keys_to_remove:
        st.session_state.pop(
            key,
            None
        )

    st.session_state.loss_file_hash = current_hash


# ============================================================
# WORKBOOK TYPE
# ============================================================

sheet_names = get_sheet_names(
    file_bytes
)

is_cluster = (
    "Fixed-CL1"
    in sheet_names
)


if is_cluster:

    ghi_columns = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI"
    ]

    input_columns = ghi_columns + [
        "Actual"
    ]

else:

    ghi_columns = [
        "GHI_Forecast"
    ]

    input_columns = [
        "GHI_Forecast",
        "Actual"
    ]


# ============================================================
# LOAD INPUT DATA
# ============================================================

input_sheet = (
    "Fixed-CL1"
    if is_cluster
    else "Fixed"
)

raw_input = get_excel_sheet(
    file_bytes,
    input_sheet,
    header=1
).copy()

raw_input.columns = (
    raw_input.columns
    .astype(str)
    .str.strip()
)

raw_input = clean_until_empty(
    raw_input,
    "Date"
)

raw_input = raw_input.iloc[
    :96
].copy()

for col in input_columns:

    if col not in raw_input.columns:

        st.error(
            f"Required column '{col}' "
            f"was not found in {input_sheet}."
        )

        st.stop()


input_df = raw_input[
    input_columns
].copy()

input_df = input_df.fillna(0)


# ============================================================
# INPUT DATA EDITOR
# ============================================================

st.subheader("📊 Input Data")

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="loss_editor"
)

edited_df = edited_df.copy()

for col in input_columns:

    edited_df[col] = pd.to_numeric(
        edited_df[col],
        errors="coerce"
    ).fillna(0)

edited_df = edited_df.iloc[
    :96
].reset_index(drop=True)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.pills(
    "Plant Type",
    [
        "🏗️ Fixed",
        "🔄 Tracking"
    ],
    default="🏗️ Fixed"
)


# ============================================================
# CALCULATE BUTTON
# ============================================================

if not st.session_state.get(
    "loss_calculated",
    False
):

    st.info(
        "Click Calculate Loss Correction once. "
        "After optimization, all parameters can be edited "
        "without running the optimization again."
    )

    calculate_button = st.button(
        "🚀 Calculate Loss Correction",
        use_container_width=True,
        type="primary"
    )

else:

    calculate_button = False


# ============================================================
# OPTIMIZATION
# ============================================================

if calculate_button:

    # --------------------------------------------------------
    # Area & Efficiency
    # --------------------------------------------------------

    df = prepare_area_efficiency(
        file_bytes,
        is_cluster
    )

    latitude = get_latitude(
        file_bytes
    )

    # --------------------------------------------------------
    # Prepare input data
    # --------------------------------------------------------

    edited_ghi = edited_df[
        ghi_columns
    ].copy()

    edited_actual = edited_df[
        ["Actual"]
    ].copy()

    df_fix, ghi_columns = (
        prepare_fixed_ghi_data(
            file_bytes,
            edited_ghi,
            edited_actual,
            is_cluster
        )
    )

    # --------------------------------------------------------
    # Cluster weights
    # --------------------------------------------------------

    cluster_weights = None

    if is_cluster:

        df_w = get_excel_sheet(
            file_bytes,
            "Area & Efficiency",
            header=2,
            usecols=[12, 13, 14, 15, 16]
        )

        cluster_weights = np.array([
            pd.to_numeric(
                df_w["CL-1"].iloc[0],
                errors="coerce"
            ),
            pd.to_numeric(
                df_w["CL-2"].iloc[0],
                errors="coerce"
            ),
            pd.to_numeric(
                df_w["CL-3"].iloc[0],
                errors="coerce"
            ),
            pd.to_numeric(
                df_w["CL-4"].iloc[0],
                errors="coerce"
            ),
            pd.to_numeric(
                df_w["CL-5"].iloc[0],
                errors="coerce"
            )
        ])

        cluster_weights = np.nan_to_num(
            cluster_weights
        )

    # --------------------------------------------------------
    # Efficiency loss
    # --------------------------------------------------------

    with st.spinner(
        "Finding the optimum efficiency loss..."
    ):

        best_loss = calculate_efficiency_loss(
            df=df,
            df_fix=df_fix,
            ghi_columns=ghi_columns,
            is_cluster=is_cluster,
            cluster_weights=cluster_weights
        )

    # --------------------------------------------------------
    # Fixed plant
    # --------------------------------------------------------

    if plant_type == "🏗️ Fixed":

        month_lookup = None

        if not is_cluster:

            df_tilt = get_excel_sheet(
                file_bytes,
                "Config Tilt Angle",
                header=7
            )

            df_tilt.columns = (
                df_tilt.columns
                .astype(str)
                .str.strip()
            )

            if "Fixed" in df_tilt.columns:

                df_tilt["Fixed"] = pd.to_numeric(
                    df_tilt["Fixed"],
                    errors="coerce"
                ).fillna(0)

                df_tilt = clean_until_empty(
                    df_tilt,
                    "Fixed"
                )

                df_tilt = df_tilt.rename(
                    columns={
                        "Unnamed: 2": "Month_Num",
                        "Unnamed: 3": "Month"
                    }
                )

                if (
                    "Month" in df_tilt.columns
                    and
                    "Fixed" in df_tilt.columns
                ):

                    month_lookup = (
                        df_tilt
                        .set_index("Month")[
                            "Fixed"
                        ]
                        .to_dict()
                    )

        st.session_state.optimization_result = {
            "plant_type": "Fixed",
            "loss": best_loss
        }

    # --------------------------------------------------------
    # Tracking plant
    # --------------------------------------------------------

    else:

        if is_cluster:

            backend_sheet = (
                "Backend Cal CL1"
            )

        else:

            backend_sheet = (
                "Backend Cal"
            )

        df_bcal = get_excel_sheet(
            file_bytes,
            backend_sheet
        )

        backend_blocks = (
            df_bcal[
                "Block No."
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # ----------------------------------------------------
        # Tracking optimization objective
        # ----------------------------------------------------

        actual = (
            df_fix["Actual"]
            .to_numpy(
                dtype=np.float64
            )
        )

        ghi = (
            df_fix["GHI_Forecast"]
            .to_numpy(
                dtype=np.float64
            )
        )

        valid_mask = (
            np.isfinite(actual)
            &
            np.isfinite(ghi)
        )

        actual_valid = actual[
            valid_mask
        ]

        ghi_valid = ghi[
            valid_mask
        ]

        effective_area = (
            df["Total area(m2)"]
            * df["Standard PV Efficiency (%)"]
            / 100
        ).sum()

        def tracking_objective(x):

            DHI = int(round(x[0]))
            start_block = int(round(x[1]))
            end_block = int(round(x[2]))
            max_block = int(round(x[3]))
            east_limit = int(round(x[4]))
            west_limit = int(round(x[5]))

            if not (
                start_block
                < max_block
                < end_block
            ):
                return 1e9

            d1 = (
                start_block
                - 1
                - max_block
            )

            d2 = (
                end_block
                + 1
                - max_block
            )

            if d1 == 0 or d2 == 0:
                return 1e9

            m1 = 90 / d1
            m2 = 90 / d2

            zenith = np.where(
                backend_blocks <= max_block,

                np.minimum(
                    89,
                    m1 * (
                        backend_blocks
                        - max_block
                    )
                ),

                np.minimum(
                    89,
                    m2 * (
                        backend_blocks
                        - max_block
                    )
                )
            )

            panel = np.where(
                backend_blocks < max_block,

                np.minimum(
                    zenith,
                    abs(east_limit)
                ),

                np.where(
                    (
                        (backend_blocks > max_block)
                        &
                        (
                            zenith
                            > west_limit
                        )
                    ),
                    west_limit,
                    zenith
                )
            )

            cos_alpha = np.cos(
                np.radians(panel)
            )

            cos_alpha = np.clip(
                cos_alpha,
                1e-6,
                None
            )

            dhi = (
                ghi_valid
                * DHI
                / 100
            )

            dni = (
                ghi_valid
                - dhi
            ) / cos_alpha[
                valid_mask
            ]

            prediction = (
                dni
                * effective_area
                / 1_000_000
            )

            scale = max(
                np.max(
                    np.abs(
                        actual_valid
                    )
                ),
                1e-6
            )

            score = np.mean(
                np.abs(
                    actual_valid
                    - prediction
                )
            ) / scale

            if not np.isfinite(score):
                return 1e9

            return score

        # ----------------------------------------------------
        # Tracking bounds
        # ----------------------------------------------------

        bounds = [
            (0, 10),      # DHI
            (0, 30),      # Starting Block
            (65, 80),     # Ending Block
            (44, 60),     # Max Block
            (0, 70),      # East Limit
            (0, 70)       # West Limit
        ]

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            result = differential_evolution(
                tracking_objective,
                bounds=bounds,
                strategy="best1bin",
                maxiter=40,
                popsize=10,
                tol=0.002,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                polish=True,
                workers=1
            )

        best = np.round(
            result.x
        ).astype(int)

        st.session_state.optimization_result = {

            "plant_type": "Tracking",

            "loss": float(best_loss),

            "DHI": int(best[0]),
            "start": int(best[1]),
            "end": int(best[2]),
            "max": int(best[3]),
            "east": int(best[4]),
            "west": int(best[5]),

            "backend_blocks": backend_blocks.tolist()
        }

    # --------------------------------------------------------
    # Save calculated state
    # --------------------------------------------------------

    st.session_state.loss_calculated = True

    st.rerun()


# ============================================================
# RESULTS
# ============================================================

if not st.session_state.get(
    "loss_calculated",
    False
):

    st.stop()


result = st.session_state.optimization_result


# ============================================================
# IMPORTANT:
# Keep widgets outside optimization.
#
# Therefore changing these values will NEVER rerun
# Differential Evolution.
# ============================================================


st.divider()

st.subheader(
    "⚙️ Correction Parameters"
)


# ============================================================
# FIXED
# ============================================================

if result["plant_type"] == "Fixed":

    # --------------------------------------------------------
    # Efficiency loss
    # --------------------------------------------------------

    if "editable_loss" not in st.session_state:

        st.session_state.editable_loss = (
            result["loss"]
        )

    best_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        key="editable_loss"
    )

    st.caption(
        "You can change the efficiency loss after "
        "optimization. The forecast updates instantly."
    )


# ============================================================
# TRACKING
# ============================================================

else:

    if "editable_loss" not in st.session_state:

        st.session_state.editable_loss = (
            result["loss"]
        )

    if "editable_dhi" not in st.session_state:

        st.session_state.editable_dhi = (
            result["DHI"]
        )

    if "editable_start" not in st.session_state:

        st.session_state.editable_start = (
            result["start"]
        )

    if "editable_end" not in st.session_state:

        st.session_state.editable_end = (
            result["end"]
        )

    if "editable_max" not in st.session_state:

        st.session_state.editable_max = (
            result["max"]
        )

    if "editable_east" not in st.session_state:

        st.session_state.editable_east = (
            result["east"]
        )

    if "editable_west" not in st.session_state:

        st.session_state.editable_west = (
            result["west"]
        )

    # --------------------------------------------------------
    # Efficiency Loss
    # --------------------------------------------------------

    best_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        key="editable_loss"
    )

    # --------------------------------------------------------
    # Tracking Parameters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    DHI = col1.number_input(
        "DHI (%)",
        min_value=0,
        max_value=100,
        step=1,
        key="editable_dhi"
    )

    GHI_Starting_Block = col2.number_input(
        "Starting Block",
        min_value=0,
        max_value=95,
        step=1,
        key="editable_start"
    )

    GHI_Ending_Block = col3.number_input(
        "Ending Block",
        min_value=1,
        max_value=96,
        step=1,
        key="editable_end"
    )

    col1, col2, col3 = st.columns(3)

    GHI_Max_Block = col1.number_input(
        "Max Block",
        min_value=0,
        max_value=96,
        step=1,
        key="editable_max"
    )

    Tracking_angle_lim_E = col2.number_input(
        "East Limit",
        min_value=0,
        max_value=90,
        step=1,
        key="editable_east"
    )

    Tracking_angle_lim_W = col3.number_input(
        "West Limit",
        min_value=0,
        max_value=90,
        step=1,
        key="editable_west"
    )

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):

        st.error(
            "Invalid tracking parameters. "
            "Required: Starting Block < Max Block < Ending Block."
        )

        st.stop()


# ============================================================
# RELOAD AREA & EFFICIENCY
# ============================================================

df = prepare_area_efficiency(
    file_bytes,
    is_cluster
)

latitude = get_latitude(
    file_bytes
)

edited_ghi = edited_df[
    ghi_columns
].copy()

edited_actual = edited_df[
    ["Actual"]
].copy()

df_fix, ghi_columns = (
    prepare_fixed_ghi_data(
        file_bytes,
        edited_ghi,
        edited_actual,
        is_cluster
    )
)


# ============================================================
# FINAL CALCULATION
# ============================================================

if result["plant_type"] == "Fixed":

    # --------------------------------------------------------
    # Tilt lookup for fixed plant
    # --------------------------------------------------------

    month_lookup = None

    if not is_cluster:

        df_tilt = get_excel_sheet(
            file_bytes,
            "Config Tilt Angle",
            header=7
        ).copy()

        df_tilt.columns = (
            df_tilt.columns
            .astype(str)
            .str.strip()
        )

        if "Fixed" in df_tilt.columns:

            df_tilt["Fixed"] = pd.to_numeric(
                df_tilt["Fixed"],
                errors="coerce"
            ).fillna(0)

            df_tilt = clean_until_empty(
                df_tilt,
                "Fixed"
            )

            df_tilt = df_tilt.rename(
                columns={
                    "Unnamed: 2": "Month_Num",
                    "Unnamed: 3": "Month"
                }
            )

            if (
                "Month" in df_tilt.columns
                and
                "Fixed" in df_tilt.columns
            ):

                month_lookup = (
                    df_tilt
                    .set_index("Month")[
                        "Fixed"
                    ]
                    .to_dict()
                )

    df, df_fix, forecast = calculate_fixed_forecast(
        df=df,
        df_fix=df_fix,
        best_loss=best_loss,
        latitude=latitude,
        month_lookup=month_lookup
    )


else:

    backend_blocks = np.asarray(
        result["backend_blocks"],
        dtype=np.float64
    )

    df, forecast = calculate_tracking_forecast(
        df=df,
        df_fix=df_fix,
        best_loss=best_loss,
        DHI=DHI,
        start_block=GHI_Starting_Block,
        end_block=GHI_Ending_Block,
        max_block=GHI_Max_Block,
        east_limit=Tracking_angle_lim_E,
        west_limit=Tracking_angle_lim_W,
        backend_blocks=backend_blocks
    )


# ============================================================
# CURRENT EFFICIENCY LOSS
# ============================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Efficiency Loss",
        f"{best_loss:.2f}%"
    )

with col2:

    st.metric(
        "Plant Type",
        (
            "Tracking"
            if result["plant_type"] == "Tracking"
            else "Fixed"
        )
    )


# ============================================================
# EFFICIENCY CALCULATION TABLE
# ============================================================

display_df = df[
    [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)"
    ]
].copy()

numeric_columns = (
    display_df
    .select_dtypes(
        include="number"
    )
    .columns
)

display_df[
    numeric_columns
] = display_df[
    numeric_columns
].round(2)


with st.expander(
    "🔍 View Efficiency Calculations"
):

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECAST VS ACTUAL
# ============================================================

st.subheader(
    "📈 Forecast vs Actual Power"
)

actual = df_fix[
    "Actual"
].to_numpy(
    dtype=np.float64
)

x = np.arange(
    1,
    len(forecast) + 1
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=forecast,
        mode="lines",
        name="Forecast",
        line=dict(
            color="#2563EB",
            width=3
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(
            color="#DC2626",
            width=3
        )
    )
)

fig.update_layout(
    title="Forecast vs Actual Power",
    template="plotly_white",
    height=500,
    hovermode="x unified",

    xaxis=dict(
        title="15 Minute Block",
        dtick=4
    ),

    yaxis=dict(
        title="Power (MW)"
    ),

    legend=dict(
        orientation="h",
        y=1.08,
        x=0
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TRACKING PARAMETER SUMMARY
# ============================================================

if result["plant_type"] == "Tracking":

    st.subheader(
        "🎯 Current Tracking Parameters"
    )

    parameter_df = pd.DataFrame({
        "Parameter": [
            "DHI (%)",
            "Starting Block",
            "Ending Block",
            "Max Block",
            "East Limit",
            "West Limit"
        ],

        "Value": [
            DHI,
            GHI_Starting_Block,
            GHI_Ending_Block,
            GHI_Max_Block,
            Tracking_angle_lim_E,
            Tracking_angle_lim_W
        ]
    })

    st.dataframe(
        parameter_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style='text-align:center;color:#9ca3af;font-size:13px'>
    Solar Loss Correction • Developed and Maintained by
    <b>Manjot Singh</b>
    </div>
    """,
    unsafe_allow_html=True
)
