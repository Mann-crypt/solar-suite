# ============================================================
# SOLAR MODULE -> INVERTER CLUSTER DISTRIBUTION
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Cluster Distribution",
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
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Cluster Distribution</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Distribute complete module types across inverter clusters
    according to inverter proportion.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "manual_df" not in st.session_state:
    st.session_state.manual_df = pd.DataFrame({
        "Module Type": [""],
        "(Wp)": [0.0],
        "Std PV Eff(%)": [0.0],
        "No of Modules": [0],
        "Area of 1 module": [0.0],
        "Total area(m2)": [0.0]
    })


# ============================================================
# 1. GENERAL INPUTS
# ============================================================

st.markdown(
    '<div class="section-title">1. Plant Inputs</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    cluster_count = st.number_input(
        "Number of Clusters",
        min_value=1,
        max_value=100,
        value=3,
        step=1
    )


with col2:

    total_ac = st.number_input(
        "Total AC",
        min_value=0.0,
        value=304.92,
        step=0.01,
        format="%.2f",
        help="Total plant AC. This will be distributed according to inverter proportion."
    )


# ============================================================
# 2. INVERTER INPUT
# ============================================================

st.markdown(
    '<div class="section-title">2. Inverter Distribution</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the number of inverters in each cluster. "
    "AC and module/DC distribution will follow these proportions."
)


inverter_cols = st.columns(
    min(int(cluster_count), 4)
)

inverter_distribution = []


for i in range(int(cluster_count)):

    col = inverter_cols[i % len(inverter_cols)]

    with col:

        inverter_count = st.number_input(
            f"Cluster {i + 1} Inverters",
            min_value=1,
            value=100,
            step=1,
            key=f"inverters_{i}"
        )

        inverter_distribution.append(
            int(inverter_count)
        )


# ============================================================
# INVERTER VALIDATION
# ============================================================

total_inverters = sum(
    inverter_distribution
)


if total_inverters <= 0:

    st.error(
        "Total number of inverters must be greater than zero."
    )

    st.stop()


cluster_proportions = (
    np.array(inverter_distribution, dtype=float)
    / total_inverters
)


cluster_names = [
    f"Cluster {i + 1}"
    for i in range(int(cluster_count))
]


# ============================================================
# SHOW INVERTER SUMMARY
# ============================================================

inverter_summary = pd.DataFrame({

    "Cluster": cluster_names,

    "Inverters": inverter_distribution,

    "Inverter Proportion (%)":
        cluster_proportions * 100,

    "AC Allocation":
        total_ac * cluster_proportions

})


st.dataframe(
    inverter_summary.style.format({
        "Inverter Proportion (%)": "{:.2f}%",
        "AC Allocation": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 3. MODULE DATA INPUT METHOD
# ============================================================

st.markdown(
    '<div class="section-title">3. Module Data</div>',
    unsafe_allow_html=True
)


input_method = st.radio(
    "Choose module data input method",
    [
        "Upload DataFrame",
        "Manual Entry"
    ],
    horizontal=True
)


# ============================================================
# UPLOAD DATAFRAME
# ============================================================

uploaded_file = None


if input_method == "Upload DataFrame":

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file",
        type=["xlsx", "xls", "csv"],
        help="Upload the file containing module type and module information."
    )

    if uploaded_file is None:

        st.warning(
            "Please upload an Excel or CSV file."
        )

        st.stop()


    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            f"Could not read the file: {e}"
        )

        st.stop()


# ============================================================
# MANUAL ENTRY
# ============================================================

else:

    st.write(
        "Enter one row for each module type."
    )

    st.caption(
        "A module type will be assigned completely to one cluster. "
        "It will never be split between clusters."
    )


    manual_columns = [
        "Module Type",
        "(Wp)",
        "Std PV Eff(%)",
        "No of Modules",
        "Area of 1 module",
        "Total area(m2)"
    ]


    df = st.data_editor(
        st.session_state.manual_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={

            "Module Type": st.column_config.TextColumn(
                "Module Type"
            ),

            "(Wp)": st.column_config.NumberColumn(
                "(Wp)",
                min_value=0,
                format="%.2f"
            ),

            "Std PV Eff(%)": st.column_config.NumberColumn(
                "Std PV Eff(%)",
                min_value=0,
                max_value=100,
                format="%.2f"
            ),

            "No of Modules": st.column_config.NumberColumn(
                "No of Modules",
                min_value=0,
                step=1
            ),

            "Area of 1 module": st.column_config.NumberColumn(
                "Area of 1 module",
                min_value=0,
                format="%.4f"
            ),

            "Total area(m2)": st.column_config.NumberColumn(
                "Total area(m2)",
                min_value=0,
                format="%.2f"
            )
        }
    )

    st.session_state.manual_df = df.copy()


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Module Type",
    "(Wp)",
    "No of Modules",
    "Area of 1 module"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Required columns are: "
        + ", ".join(required_columns)
    )

    st.stop()


# ============================================================
# 5. CLEAN DATA
# ============================================================

df = df.copy()


# Remove completely empty rows

df = df.dropna(
    how="all"
).reset_index(
    drop=True
)


# Remove rows without module type

df["Module Type"] = (
    df["Module Type"]
    .astype(str)
    .str.strip()
)


df = df[
    (df["Module Type"] != "")
    &
    (df["Module Type"].str.lower() != "nan")
].copy()


if df.empty:

    st.warning(
        "No valid module data found."
    )

    st.stop()


# ============================================================
# 6. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "(Wp)",
    "No of Modules",
    "Area of 1 module"
]


if "Std PV Eff(%)" in df.columns:

    numeric_columns.append(
        "Std PV Eff(%)"
    )


if "Total area(m2)" in df.columns:

    numeric_columns.append(
        "Total area(m2)"
    )


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# VALIDATE NUMERIC DATA
# ============================================================

if df["(Wp)"].isna().any():

    st.error(
        "Some rows have invalid or missing Wp values."
    )

    st.stop()


if df["No of Modules"].isna().any():

    st.error(
        "Some rows have invalid or missing No of Modules values."
    )

    st.stop()


if (
    df["No of Modules"] < 0
).any():

    st.error(
        "Number of modules cannot be negative."
    )

    st.stop()


# ============================================================
# 7. HANDLE DUPLICATE MODULE TYPES
# ============================================================
#
# We do NOT want the same module type to appear in multiple
# clusters.
#
# If the uploaded file contains the same module type multiple
# times, combine those rows first.
#
# ============================================================

duplicate_types = (
    df["Module Type"]
    .duplicated(keep=False)
)


if duplicate_types.any():

    duplicate_names = (
        df.loc[
            duplicate_types,
            "Module Type"
        ]
        .unique()
        .tolist()
    )

    st.warning(
        "Duplicate module types detected. "
        "Rows with the same module type will be combined "
        "before cluster allocation."
    )

    aggregation = {}

    for col in df.columns:

        if col == "Module Type":
            continue

        elif col in [
            "No of Modules",
            "Total area(m2)"
        ]:

            aggregation[col] = "sum"

        else:

            aggregation[col] = "first"


    df = (
        df
        .groupby(
            "Module Type",
            as_index=False
        )
        .agg(aggregation)
    )


# ============================================================
# 8. CALCULATE TOTAL AREA IF NECESSARY
# ============================================================

df["Total area(m2)"] = (
    df["Area of 1 module"]
    *
    df["No of Modules"]
)


# ============================================================
# 9. CALCULATE DC
# ============================================================

df["Calculated DC (MW)"] = (
    df["(Wp)"]
    *
    df["No of Modules"]
) / 1e6


# ============================================================
# 10. TOTAL PLANT DC
# ============================================================

total_dc = df[
    "Calculated DC (MW)"
].sum()


# ============================================================
# 11. PLANT METRICS
# ============================================================

metric1, metric2, metric3 = st.columns(3)


with metric1:

    st.metric(
        "Total AC",
        f"{total_ac:,.2f}"
    )


with metric2:

    st.metric(
        "Total DC",
        f"{total_dc:,.2f} MW"
    )


with metric3:

    st.metric(
        "Module Types",
        f"{len(df):,}"
    )


# ============================================================
# 12. MODULE DATA PREVIEW
# ============================================================

with st.expander(
    "View calculated module data",
    expanded=False
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 13. CALCULATE TARGET DC
# ============================================================

target_dc = (
    total_dc
    *
    cluster_proportions
)


# ============================================================
# 14. DISTRIBUTION ALGORITHM
# ============================================================
#
# IMPORTANT:
#
# Each module type is assigned completely to ONE cluster.
#
# We sort module types by DC from largest to smallest.
#
# For every module type:
#
#   1. Calculate current remaining DC of every cluster.
#   2. Select the cluster with the largest remaining target.
#   3. Assign the COMPLETE module type to that cluster.
#
# No module type is duplicated.
#
# ============================================================

module_df = (
    df
    .sort_values(
        "Calculated DC (MW)",
        ascending=False
    )
    .reset_index(drop=True)
)


cluster_current_dc = np.zeros(
    int(cluster_count),
    dtype=float
)


assigned_clusters = []


for _, row in module_df.iterrows():

    remaining_dc = (
        target_dc
        -
        cluster_current_dc
    )


    # Prefer clusters that have not reached
    # their target DC.

    available_clusters = np.where(
        remaining_dc >= 0
    )[0]


    if len(available_clusters) > 0:

        selected_cluster = (
            available_clusters[
                np.argmax(
                    remaining_dc[
                        available_clusters
                    ]
                )
            ]
        )

    else:

        # If all targets have been exceeded,
        # put the module type into the cluster
        # with the lowest current DC.

        selected_cluster = int(
            np.argmin(
                cluster_current_dc
            )
        )


    assigned_clusters.append(
        cluster_names[
            selected_cluster
        ]
    )


    cluster_current_dc[
        selected_cluster
    ] += row[
        "Calculated DC (MW)"
    ]


# ============================================================
# 15. CREATE DISTRIBUTED DATAFRAME
# ============================================================

module_df["Cluster"] = (
    assigned_clusters
)


# ============================================================
# 16. ORDER COLUMNS
# ============================================================

output_columns = [
    "Cluster",
    "Module Type",
    "(Wp)",
    "Std PV Eff(%)",
    "No of Modules",
    "Area of 1 module",
    "Total area(m2)",
    "Calculated DC (MW)"
]


output_columns = [
    col
    for col in output_columns
    if col in module_df.columns
]


distributed_df = module_df[
    output_columns
].copy()


# ============================================================
# 17. SORT BY CLUSTER
# ============================================================

cluster_order = {
    cluster_names[i]: i
    for i in range(int(cluster_count))
}


distributed_df["_order"] = (
    distributed_df["Cluster"]
    .map(cluster_order)
)


distributed_df = (
    distributed_df
    .sort_values(
        ["_order", "Calculated DC (MW)"],
        ascending=[True, False]
    )
    .drop(
        columns="_order"
    )
    .reset_index(drop=True)
)


# ============================================================
# 18. CALCULATE ACTUAL CLUSTER DC
# ============================================================

actual_cluster_dc = (
    distributed_df
    .groupby(
        "Cluster",
        sort=False
    )["Calculated DC (MW)"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0
    )
    .values
)


# ============================================================
# 19. AC DISTRIBUTION
# ============================================================

cluster_ac = (
    total_ac
    *
    cluster_proportions
)


# ============================================================
# 20. CLUSTER SUMMARY
# ============================================================

cluster_summary = pd.DataFrame({

    "Cluster": cluster_names,

    "Inverters": inverter_distribution,

    "Inverter Proportion (%)":
        cluster_proportions * 100,

    "AC":
        cluster_ac,

    "Target DC (MW)":
        target_dc,

    "Actual DC (MW)":
        actual_cluster_dc,

    "DC Difference (MW)":
        actual_cluster_dc - target_dc

})


# ============================================================
# 21. MODULE COUNTS
# ============================================================

module_count_summary = (
    distributed_df
    .groupby(
        "Cluster",
        sort=False
    )["No of Modules"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0
    )
)


cluster_summary[
    "No of Modules"
] = (
    module_count_summary.values
)


# ============================================================
# 22. MODULE TYPE COUNTS
# ============================================================

module_type_summary = (
    distributed_df
    .groupby(
        "Cluster",
        sort=False
    )["Module Type"]
    .count()
    .reindex(
        cluster_names,
        fill_value=0
    )
)


cluster_summary[
    "Module Types"
] = (
    module_type_summary.values
)


# ============================================================
# 23. AREA
# ============================================================

area_summary = (
    distributed_df
    .groupby(
        "Cluster",
        sort=False
    )["Total area(m2)"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0
    )
)


cluster_summary[
    "Total Area (m2)"
] = (
    area_summary.values
)


# ============================================================
# 24. FINAL COLUMN ORDER
# ============================================================

cluster_summary = cluster_summary[
    [
        "Cluster",
        "Inverters",
        "Inverter Proportion (%)",
        "AC",
        "Module Types",
        "No of Modules",
        "Total Area (m2)",
        "Target DC (MW)",
        "Actual DC (MW)",
        "DC Difference (MW)"
    ]
]


# ============================================================
# 25. RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">4. Distribution Results</div>',
    unsafe_allow_html=True
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

st.subheader(
    "Cluster Summary"
)


st.dataframe(
    cluster_summary.style.format({

        "Inverter Proportion (%)":
            "{:.2f}%",

        "AC":
            "{:.2f}",

        "Total Area (m2)":
            "{:,.2f}",

        "Target DC (MW)":
            "{:,.4f}",

        "Actual DC (MW)":
            "{:,.4f}",

        "DC Difference (MW)":
            "{:+,.4f}"

    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODULE DISTRIBUTION
# ============================================================

st.subheader(
    "Module Type → Cluster Distribution"
)


st.caption(
    "Each module type appears exactly once. "
    "No module type is split between clusters."
)


st.dataframe(
    distributed_df.style.format({

        "(Wp)": "{:,.2f}",

        "Std PV Eff(%)": "{:.2f}",

        "No of Modules": "{:,.0f}",

        "Area of 1 module": "{:,.4f}",

        "Total area(m2)": "{:,.2f}",

        "Calculated DC (MW)": "{:,.6f}"

    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 26. VALIDATION
# ============================================================

st.markdown(
    '<div class="section-title">5. Validation</div>',
    unsafe_allow_html=True
)


original_dc = (
    df["Calculated DC (MW)"].sum()
)


distributed_dc = (
    distributed_df["Calculated DC (MW)"].sum()
)


original_module_count = (
    df["No of Modules"].sum()
)


distributed_module_count = (
    distributed_df["No of Modules"].sum()
)


original_type_count = (
    df["Module Type"].nunique()
)


distributed_type_count = (
    distributed_df["Module Type"].nunique()
)


duplicate_after_distribution = (
    distributed_df[
        "Module Type"
    ].duplicated(
        keep=False
    )
    .any()
)


validation1, validation2, validation3, validation4 = st.columns(4)


with validation1:

    if np.isclose(
        original_dc,
        distributed_dc,
        atol=1e-9
    ):

        st.success(
            "✓ DC preserved"
        )

    else:

        st.error(
            "✗ DC mismatch"
        )


with validation2:

    if np.isclose(
        original_module_count,
        distributed_module_count
    ):

        st.success(
            "✓ Modules preserved"
        )

    else:

        st.error(
            "✗ Module mismatch"
        )


with validation3:

    if (
        original_type_count
        ==
        distributed_type_count
    ):

        st.success(
            "✓ Module types preserved"
        )

    else:

        st.error(
            "✗ Module type mismatch"
        )


with validation4:

    if not duplicate_after_distribution:

        st.success(
            "✓ No duplicate module types"
        )

    else:

        st.error(
            "✗ Duplicate module types"
        )


# ============================================================
# 27. DOWNLOAD RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">6. Download Results</div>',
    unsafe_allow_html=True
)


excel_buffer = io.BytesIO()


with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl"
) as writer:

    distributed_df.to_excel(
        writer,
        sheet_name="Module Distribution",
        index=False
    )

    cluster_summary.to_excel(
        writer,
        sheet_name="Cluster Summary",
        index=False
    )

    df.to_excel(
        writer,
        sheet_name="Original Module Data",
        index=False
    )


st.download_button(
    label="⬇️ Download Excel Result",
    data=excel_buffer.getvalue(),
    file_name="Solar_Cluster_Distribution.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True
)


# ============================================================
# 28. DOWNLOAD CSV
# ============================================================

csv_data = distributed_df.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Module Distribution CSV",
    data=csv_data,
    file_name="Module_Cluster_Distribution.csv",
    mime="text/csv",
    use_container_width=True
)
