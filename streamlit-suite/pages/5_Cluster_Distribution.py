# ============================================================
# SOLAR MODULE -> INVERTER CLUSTER DISTRIBUTION
# Stable Streamlit Version
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
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("☀️ Solar Module Cluster Distribution")

st.caption(
    "Distribute complete module types to inverter clusters "
    "according to inverter proportion."
)


# ============================================================
# STEP 1: CLUSTERS
# ============================================================

st.header("1. Cluster Information")

cluster_count = st.number_input(
    "Enter number of clusters",
    min_value=1,
    max_value=100,
    step=1,
    value=None,
    placeholder="Enter cluster count",
)


if cluster_count is None:
    st.info("Enter the number of clusters to continue.")
    st.stop()


cluster_count = int(cluster_count)


# ============================================================
# STEP 2: INVERTERS
# ============================================================

st.header("2. Inverter Distribution")

st.write(
    "Enter the number of inverters assigned to each cluster."
)

inverter_distribution = []

inverter_input_complete = True


for i in range(cluster_count):

    inverter_value = st.number_input(
        f"Cluster {i + 1} - Number of Inverters",
        min_value=1,
        max_value=1000000,
        step=1,
        value=None,
        placeholder="Enter inverter count",
        key=f"cluster_inverter_{i}",
    )

    if inverter_value is None:

        inverter_input_complete = False
        inverter_distribution.append(0)

    else:

        inverter_distribution.append(
            int(inverter_value)
        )


if not inverter_input_complete:

    st.info(
        "Enter the inverter count for every cluster."
    )

    st.stop()


total_inverters = sum(
    inverter_distribution
)


if total_inverters <= 0:

    st.error(
        "Total inverter count must be greater than zero."
    )

    st.stop()


# ============================================================
# CLUSTER PROPORTIONS
# ============================================================

cluster_names = [
    f"Cluster {i + 1}"
    for i in range(cluster_count)
]


cluster_proportions = (
    np.asarray(
        inverter_distribution,
        dtype=float,
    )
    / float(total_inverters)
)


# ============================================================
# STEP 3: AC
# ============================================================

st.header("3. AC Input")

total_ac = st.number_input(
    "Enter total plant AC",
    min_value=0.0,
    step=0.01,
    value=None,
    placeholder="Enter total AC",
)


if total_ac is None:

    st.info(
        "Enter total plant AC to continue."
    )

    st.stop()


total_ac = float(total_ac)


# ============================================================
# AC DISTRIBUTION
# ============================================================

cluster_ac = (
    total_ac
    * cluster_proportions
)


# ============================================================
# AC SUMMARY
# ============================================================

ac_summary = pd.DataFrame(
    {
        "Cluster": cluster_names,
        "Inverters": inverter_distribution,
        "Inverter Share (%)": (
            cluster_proportions * 100
        ),
        "AC": cluster_ac,
    }
)


st.subheader("AC Distribution")

st.dataframe(
    ac_summary.style.format(
        {
            "Inverter Share (%)": "{:.2f}%",
            "AC": "{:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# STEP 4: MODULE DATA
# ============================================================

st.header("4. Module Data")

input_method = st.radio(
    "Select module data input method",
    [
        "Upload File",
        "Manual Entry",
    ],
    index=None,
)


if input_method is None:

    st.info(
        "Select Upload File or Manual Entry."
    )

    st.stop()


# ============================================================
# UPLOAD FILE
# ============================================================

if input_method == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload module data",
        type=[
            "xlsx",
            "xls",
            "csv",
        ],
    )


    if uploaded_file is None:

        st.info(
            "Upload an Excel or CSV file to continue."
        )

        st.stop()


    try:

        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        elif (
            filename.endswith(".xlsx")
            or filename.endswith(".xls")
        ):

            df = pd.read_excel(
                uploaded_file
            )

        else:

            st.error(
                "Unsupported file format."
            )

            st.stop()


    except Exception as error:

        st.error(
            f"Unable to read the uploaded file: {error}"
        )

        st.stop()


# ============================================================
# MANUAL ENTRY
# ============================================================

else:

    st.write(
        "Enter one row for every module type."
    )

    st.caption(
        "Each module type must appear only once."
    )


    module_type_count = st.number_input(
        "Enter number of module types",
        min_value=1,
        max_value=10000,
        step=1,
        value=None,
        placeholder="Enter number of module types",
    )


    if module_type_count is None:

        st.info(
            "Enter the number of module types."
        )

        st.stop()


    module_type_count = int(
        module_type_count
    )


    manual_rows = []

    manual_input_complete = True


    for i in range(module_type_count):

        st.markdown(
            f"**Module Type {i + 1}**"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            module_type = st.text_input(
                "Module Type",
                placeholder="Enter module type",
                key=f"module_type_{i}",
            )


        with col2:

            wp = st.number_input(
                "Wp",
                min_value=0.0,
                step=0.01,
                value=None,
                placeholder="Enter Wp",
                key=f"module_wp_{i}",
            )


        with col3:

            efficiency = st.number_input(
                "Std PV Eff (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                value=None,
                placeholder="Enter efficiency",
                key=f"module_efficiency_{i}",
            )


        col4, col5 = st.columns(2)


        with col4:

            module_count = st.number_input(
                "No of Modules",
                min_value=1,
                step=1,
                value=None,
                placeholder="Enter module count",
                key=f"module_count_{i}",
            )


        with col5:

            area = st.number_input(
                "Area of 1 module",
                min_value=0.0,
                step=0.0001,
                value=None,
                placeholder="Enter module area",
                key=f"module_area_{i}",
            )


        if (
            not module_type
            or wp is None
            or efficiency is None
            or module_count is None
            or area is None
        ):

            manual_input_complete = False


        manual_rows.append(
            {
                "Module Type": module_type,
                "(Wp)": wp,
                "Std PV Eff(%)": efficiency,
                "No of Modules": module_count,
                "Area of 1 module": area,
            }
        )


    if not manual_input_complete:

        st.info(
            "Complete all module type fields to continue."
        )

        st.stop()


    df = pd.DataFrame(
        manual_rows
    )


# ============================================================
# STEP 5: DATA VALIDATION
# ============================================================

st.header("5. Module Data Validation")


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [
    "Module Type",
    "(Wp)",
    "No of Modules",
    "Area of 1 module",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Required columns: "
        + ", ".join(required_columns)
    )

    st.stop()


# ============================================================
# COPY DATA
# ============================================================

df = df.copy()


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = (
    df
    .dropna(
        how="all"
    )
    .reset_index(
        drop=True
    )
)


if df.empty:

    st.error(
        "The module data contains no usable rows."
    )

    st.stop()


# ============================================================
# CLEAN MODULE TYPE
# ============================================================

df["Module Type"] = (
    df["Module Type"]
    .astype("string")
    .str.strip()
)


df = df[
    df["Module Type"].notna()
    &
    (
        df["Module Type"]
        != ""
    )
].copy()


if df.empty:

    st.error(
        "No valid Module Type values were found."
    )

    st.stop()


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "(Wp)",
    "No of Modules",
    "Area of 1 module",
]


if "Std PV Eff(%)" in df.columns:

    numeric_columns.append(
        "Std PV Eff(%)"
    )


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# INVALID NUMERIC VALUES
# ============================================================

invalid_numeric = (
    df[
        [
            "(Wp)",
            "No of Modules",
            "Area of 1 module",
        ]
    ]
    .isna()
    .any(
        axis=1
    )
)


if invalid_numeric.any():

    bad_rows = (
        df.index[
            invalid_numeric
        ]
        .tolist()
    )

    st.error(
        "Invalid or missing numeric data found."
    )

    st.write(
        "Affected row numbers:",
        bad_rows,
    )

    st.stop()


# ============================================================
# VALUE VALIDATION
# ============================================================

if (
    df["(Wp)"] <= 0
).any():

    st.error(
        "Wp must be greater than zero."
    )

    st.stop()


if (
    df["No of Modules"] <= 0
).any():

    st.error(
        "No of Modules must be greater than zero."
    )

    st.stop()


if (
    df["Area of 1 module"] < 0
).any():

    st.error(
        "Area of 1 module cannot be negative."
    )

    st.stop()


# ============================================================
# MODULE COUNT MUST BE WHOLE NUMBER
# ============================================================

module_count_values = (
    df["No of Modules"]
    .to_numpy()
)


if not np.all(
    np.isclose(
        module_count_values,
        np.round(
            module_count_values
        ),
    )
):

    st.error(
        "No of Modules must contain whole numbers."
    )

    st.stop()


df["No of Modules"] = (
    np.round(
        df["No of Modules"]
    )
    .astype("int64")
)


# ============================================================
# DUPLICATE MODULE TYPES
# ============================================================

duplicate_mask = (
    df["Module Type"]
    .duplicated(
        keep=False
    )
)


if duplicate_mask.any():

    duplicate_types = (
        df.loc[
            duplicate_mask,
            "Module Type",
        ]
        .drop_duplicates()
        .tolist()
    )


    st.error(
        "Duplicate Module Type detected."
    )


    st.write(
        "Duplicate module types:"
    )


    st.write(
        duplicate_types
    )


    st.warning(
        "Each Module Type must appear only once. "
        "Please correct the source data."
    )


    st.stop()


# ============================================================
# CALCULATE AREA
# ============================================================

df["Total area(m2)"] = (
    df["Area of 1 module"]
    *
    df["No of Modules"]
)


# ============================================================
# CALCULATE DC
# ============================================================

df["Calculated DC (MW)"] = (
    df["(Wp)"]
    *
    df["No of Modules"]
) / 1_000_000


# ============================================================
# CHECK DC
# ============================================================

if (
    df["Calculated DC (MW)"] <= 0
).any():

    st.error(
        "Calculated DC contains zero or negative values."
    )

    st.stop()


# ============================================================
# TOTAL DC
# ============================================================

total_dc = float(
    df["Calculated DC (MW)"].sum()
)


# ============================================================
# PLANT SUMMARY
# ============================================================

st.subheader("Plant Summary")


summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


with summary_col1:

    st.metric(
        "Total AC",
        f"{total_ac:,.2f}",
    )


with summary_col2:

    st.metric(
        "Calculated DC",
        f"{total_dc:,.4f} MW",
    )


with summary_col3:

    st.metric(
        "Module Types",
        f"{len(df):,}",
    )


# ============================================================
# MODULE DATA PREVIEW
# ============================================================

with st.expander(
    "View module data and calculated DC"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STEP 6: DC TARGET
# ============================================================

st.header("6. Cluster DC Targets")


target_dc = (
    total_dc
    *
    cluster_proportions
)


target_table = pd.DataFrame(
    {
        "Cluster": cluster_names,
        "Inverters": inverter_distribution,
        "Inverter Share (%)": (
            cluster_proportions * 100
        ),
        "AC": cluster_ac,
        "Target DC (MW)": target_dc,
    }
)


st.dataframe(
    target_table.style.format(
        {
            "Inverter Share (%)": "{:.2f}%",
            "AC": "{:,.2f}",
            "Target DC (MW)": "{:,.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# STEP 7: MODULE TYPE ALLOCATION
# ============================================================

st.header("7. Module Type Distribution")


st.info(
    "Each complete Module Type is assigned to exactly one "
    "cluster. No module type is split between clusters."
)


# ============================================================
# SORT LARGEST DC FIRST
# ============================================================

module_df = (
    df
    .sort_values(
        "Calculated DC (MW)",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# CURRENT CLUSTER DC
# ============================================================

current_cluster_dc = np.zeros(
    cluster_count,
    dtype=float,
)


assigned_clusters = []


# ============================================================
# ASSIGN EACH MODULE TYPE
# ============================================================

for _, row in module_df.iterrows():

    module_dc = float(
        row["Calculated DC (MW)"]
    )


    remaining_target = (
        target_dc
        -
        current_cluster_dc
    )


    # --------------------------------------------------------
    # Find clusters that are still below target
    # --------------------------------------------------------

    available = np.where(
        remaining_target > 0
    )[0]


    if len(available) > 0:

        selected_cluster = int(
            available[
                np.argmax(
                    remaining_target[
                        available
                    ]
                )
            ]
        )


    else:

        selected_cluster = int(
            np.argmin(
                current_cluster_dc
            )
        )


    # --------------------------------------------------------
    # Assign COMPLETE module type
    # --------------------------------------------------------

    assigned_clusters.append(
        cluster_names[
            selected_cluster
        ]
    )


    current_cluster_dc[
        selected_cluster
    ] += module_dc


# ============================================================
# ADD CLUSTER
# ============================================================

module_df["Cluster"] = (
    assigned_clusters
)


# ============================================================
# SORT OUTPUT
# ============================================================

cluster_order = {
    cluster_names[i]: i
    for i in range(cluster_count)
}


module_df["_cluster_order"] = (
    module_df["Cluster"]
    .map(cluster_order)
)


distributed_df = (
    module_df
    .sort_values(
        [
            "_cluster_order",
            "Calculated DC (MW)",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop(
        columns="_cluster_order"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [
    "Cluster",
    "Module Type",
    "(Wp)",
    "Std PV Eff(%)",
    "No of Modules",
    "Area of 1 module",
    "Total area(m2)",
    "Calculated DC (MW)",
]


output_columns = [
    column
    for column in output_columns
    if column in distributed_df.columns
]


distributed_df = distributed_df[
    output_columns
]


# ============================================================
# SHOW RESULT
# ============================================================

st.subheader(
    "Module Type → Cluster"
)


st.caption(
    "Each module type appears once only."
)


st.dataframe(
    distributed_df.style.format(
        {
            "(Wp)": "{:,.2f}",
            "Std PV Eff(%)": "{:,.2f}",
            "No of Modules": "{:,.0f}",
            "Area of 1 module": "{:,.4f}",
            "Total area(m2)": "{:,.2f}",
            "Calculated DC (MW)": "{:,.6f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# STEP 8: FINAL CLUSTER SUMMARY
# ============================================================

st.header("8. Final Cluster Summary")


actual_dc = (
    distributed_df
    .groupby(
        "Cluster"
    )["Calculated DC (MW)"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0,
    )
    .to_numpy()
)


module_counts = (
    distributed_df
    .groupby(
        "Cluster"
    )["No of Modules"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0,
    )
    .to_numpy()
)


module_type_counts = (
    distributed_df
    .groupby(
        "Cluster"
    )["Module Type"]
    .count()
    .reindex(
        cluster_names,
        fill_value=0,
    )
    .to_numpy()
)


area_counts = (
    distributed_df
    .groupby(
        "Cluster"
    )["Total area(m2)"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0,
    )
    .to_numpy()
)


final_summary = pd.DataFrame(
    {
        "Cluster": cluster_names,
        "Inverters": inverter_distribution,
        "Inverter Share (%)": (
            cluster_proportions * 100
        ),
        "AC": cluster_ac,
        "Module Types": module_type_counts,
        "No of Modules": module_counts,
        "Total Area (m2)": area_counts,
        "Target DC (MW)": target_dc,
        "Actual DC (MW)": actual_dc,
        "DC Difference (MW)": (
            actual_dc - target_dc
        ),
    }
)


st.dataframe(
    final_summary.style.format(
        {
            "Inverter Share (%)": "{:.2f}%",
            "AC": "{:,.2f}",
            "Total Area (m2)": "{:,.2f}",
            "Target DC (MW)": "{:,.4f}",
            "Actual DC (MW)": "{:,.4f}",
            "DC Difference (MW)": "{:+,.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# STEP 9: VALIDATION
# ============================================================

st.header("9. Validation")


original_dc = float(
    df["Calculated DC (MW)"].sum()
)


result_dc = float(
    distributed_df["Calculated DC (MW)"].sum()
)


original_modules = int(
    df["No of Modules"].sum()
)


result_modules = int(
    distributed_df["No of Modules"].sum()
)


original_module_types = int(
    df["Module Type"].nunique()
)


result_module_types = int(
    distributed_df["Module Type"].nunique()
)


duplicate_result = (
    distributed_df["Module Type"]
    .duplicated()
    .any()
)


validation_cols = st.columns(4)


with validation_cols[0]:

    if np.isclose(
        original_dc,
        result_dc,
        atol=1e-9,
    ):

        st.success(
            "✓ Total DC preserved"
        )

    else:

        st.error(
            "✗ DC mismatch"
        )


with validation_cols[1]:

    if (
        original_modules
        ==
        result_modules
    ):

        st.success(
            "✓ Modules preserved"
        )

    else:

        st.error(
            "✗ Module count mismatch"
        )


with validation_cols[2]:

    if (
        original_module_types
        ==
        result_module_types
    ):

        st.success(
            "✓ Module types preserved"
        )

    else:

        st.error(
            "✗ Module type mismatch"
        )


with validation_cols[3]:

    if not duplicate_result:

        st.success(
            "✓ No duplicate module types"
        )

    else:

        st.error(
            "✗ Duplicate module types"
        )


# ============================================================
# STEP 10: DOWNLOAD
# ============================================================

st.header("10. Download Results")


excel_buffer = io.BytesIO()


with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl",
) as writer:

    distributed_df.to_excel(
        writer,
        sheet_name="Module Distribution",
        index=False,
    )

    final_summary.to_excel(
        writer,
        sheet_name="Cluster Summary",
        index=False,
    )

    df.to_excel(
        writer,
        sheet_name="Original Data",
        index=False,
    )


st.download_button(
    label="⬇️ Download Excel",
    data=excel_buffer.getvalue(),
    file_name="Solar_Cluster_Distribution.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True,
)


# ============================================================
# CSV DOWNLOAD
# ============================================================

csv_data = distributed_df.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Module Distribution CSV",
    data=csv_data,
    file_name="Solar_Module_Cluster_Distribution.csv",
    mime="text/csv",
    use_container_width=True,
)
