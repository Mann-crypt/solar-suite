# ============================================================
# SOLAR MODULE -> INVERTER CLUSTER DISTRIBUTION
# STREAMLIT APP
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
# TITLE
# ============================================================

st.title("☀️ Solar Module → Cluster Distribution")

st.write(
    "Distribute complete module types across inverter clusters "
    "according to the inverter distribution."
)


# ============================================================
# 1. CLUSTER COUNT
# ============================================================

st.header("Cluster Configuration")

cluster_count = st.number_input(
    "Number of Clusters",
    min_value=1,
    max_value=100,
    step=1,
    value=None,
    placeholder="Enter number of clusters"
)

if cluster_count is None:
    st.info("Enter the number of clusters.")
    st.stop()

cluster_count = int(cluster_count)


# ============================================================
# 2. INVERTER DISTRIBUTION
# ============================================================

st.subheader("Inverter Distribution")

st.write(
    "Enter the number of inverters for each cluster."
)

inverter_cols = st.columns(cluster_count)

inverter_distribution = []

all_inverters_entered = True

for i in range(cluster_count):

    with inverter_cols[i]:

        value = st.number_input(
            f"Cluster {i + 1}",
            min_value=1,
            step=1,
            value=None,
            placeholder="Inverters",
            key=f"inverters_{i}"
        )

        if value is None:
            all_inverters_entered = False
            inverter_distribution.append(0)
        else:
            inverter_distribution.append(int(value))


if not all_inverters_entered:

    st.info(
        "Enter inverter count for every cluster."
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
# 3. TOTAL AC
# ============================================================

st.subheader("AC Input")

total_ac = st.number_input(
    "Total Plant AC",
    min_value=0.0,
    step=0.01,
    value=None,
    placeholder="Enter total AC"
)

if total_ac is None:

    st.info(
        "Enter total plant AC."
    )

    st.stop()


total_ac = float(total_ac)


# ============================================================
# PROPORTIONS
# ============================================================

cluster_names = [
    f"Cluster {i + 1}"
    for i in range(cluster_count)
]

inverter_proportion = (
    np.array(
        inverter_distribution,
        dtype=float
    )
    / total_inverters
)


cluster_ac = (
    total_ac
    * inverter_proportion
)


# ============================================================
# AC SUMMARY
# ============================================================

ac_summary = pd.DataFrame({
    "Cluster": cluster_names,
    "Inverters": inverter_distribution,
    "Inverter Share (%)":
        inverter_proportion * 100,
    "AC": cluster_ac
})


st.subheader("AC Distribution")

st.dataframe(
    ac_summary.style.format({
        "Inverter Share (%)": "{:.2f}%",
        "AC": "{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 4. MODULE DATA INPUT
# ============================================================

st.header("Module Data")

input_method = st.radio(
    "Select how you want to provide the module DataFrame",
    [
        "Manual Entry",
        "Upload File"
    ],
    index=None
)


if input_method is None:

    st.info(
        "Select Manual Entry or Upload File."
    )

    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Module Type",
    "(Wp)",
    "Std PV Eff(%)",
    "No of Modules",
    "Area of 1 module",
    "Total area(m2)"
]


# ============================================================
# MANUAL ENTRY
# ============================================================

if input_method == "Manual Entry":

    st.subheader("Enter Module Data")

    st.write(
        "Enter or paste your module data directly into the table."
    )

    st.caption(
        "Each Module Type should appear only once."
    )


    # --------------------------------------------------------
    # Number of rows
    # --------------------------------------------------------

    row_count = st.number_input(
        "Number of Module Type Rows",
        min_value=1,
        max_value=10000,
        step=1,
        value=None,
        placeholder="Enter number of rows"
    )


    if row_count is None:

        st.info(
            "Enter the number of module type rows."
        )

        st.stop()


    row_count = int(row_count)


    # --------------------------------------------------------
    # Empty DF
    # --------------------------------------------------------

    manual_template = pd.DataFrame({

        "Module Type":
            [""] * row_count,

        "(Wp)":
            [None] * row_count,

        "Std PV Eff(%)":
            [None] * row_count,

        "No of Modules":
            [None] * row_count,

        "Area of 1 module":
            [None] * row_count,

        "Total area(m2)":
            [None] * row_count
    })


    # --------------------------------------------------------
    # Editable DF
    # --------------------------------------------------------

    df = st.data_editor(
        manual_template,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={

            "Module Type":
                st.column_config.TextColumn(
                    "Module Type"
                ),

            "(Wp)":
                st.column_config.NumberColumn(
                    "Wp",
                    min_value=0
                ),

            "Std PV Eff(%)":
                st.column_config.NumberColumn(
                    "Std PV Eff (%)",
                    min_value=0,
                    max_value=100
                ),

            "No of Modules":
                st.column_config.NumberColumn(
                    "No of Modules",
                    min_value=0,
                    step=1
                ),

            "Area of 1 module":
                st.column_config.NumberColumn(
                    "Area of 1 module",
                    min_value=0
                ),

            "Total area(m2)":
                st.column_config.NumberColumn(
                    "Total area (m²)",
                    disabled=True
                )
        },
        key="manual_module_dataframe"
    )


# ============================================================
# UPLOAD
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload Module DataFrame",
        type=[
            "xlsx",
            "xls",
            "csv"
        ]
    )


    if uploaded_file is None:

        st.info(
            "Upload an Excel or CSV file."
        )

        st.stop()


    try:

        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as error:

        st.error(
            f"Could not read file: {error}"
        )

        st.stop()


# ============================================================
# 5. VALIDATE DATAFRAME
# ============================================================

st.header("Module Data Validation")


# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


# Total area is calculated automatically, so don't require it
missing_columns = [
    column
    for column in required_columns[:-1]
    if column not in df.columns
]


if missing_columns:

    st.error(
        "Missing required columns:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# CLEAN DF
# ============================================================

df = df.copy()

df = (
    df
    .dropna(
        how="all"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# REMOVE EMPTY MODULE TYPES
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
        df["Module Type"] != ""
    )
].copy()


if df.empty:

    st.error(
        "No module data found."
    )

    st.stop()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "(Wp)",
    "Std PV Eff(%)",
    "No of Modules",
    "Area of 1 module"
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CHECK MISSING NUMERIC DATA
# ============================================================

invalid_rows = (
    df[
        numeric_columns
    ]
    .isna()
    .any(
        axis=1
    )
)


if invalid_rows.any():

    bad_rows = (
        df.index[
            invalid_rows
        ]
        .tolist()
    )

    st.error(
        "Missing or invalid numeric values found."
    )

    st.write(
        "Affected rows:",
        [x + 1 for x in bad_rows]
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
# MODULE COUNT INTEGER CHECK
# ============================================================

module_counts = (
    df["No of Modules"]
    .to_numpy()
)


if not np.all(
    np.isclose(
        module_counts,
        np.round(module_counts)
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
    .astype(int)
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
            "Module Type"
        ]
        .drop_duplicates()
        .tolist()
    )


    st.error(
        "Duplicate Module Type detected."
    )


    st.write(
        duplicate_types
    )


    st.warning(
        "A Module Type cannot appear more than once."
    )

    st.stop()


# ============================================================
# CALCULATE TOTAL AREA
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
# TOTAL DC
# ============================================================

total_dc = float(
    df["Calculated DC (MW)"].sum()
)


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("Input Module Data")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 6. TARGET DC
# ============================================================

st.header("Cluster DC Targets")

target_dc = (
    total_dc
    *
    inverter_proportion
)


target_summary = pd.DataFrame({

    "Cluster":
        cluster_names,

    "Inverters":
        inverter_distribution,

    "Inverter Share (%)":
        inverter_proportion * 100,

    "AC":
        cluster_ac,

    "Target DC (MW)":
        target_dc
})


st.dataframe(
    target_summary.style.format({

        "Inverter Share (%)":
            "{:.2f}%",

        "AC":
            "{:,.2f}",

        "Target DC (MW)":
            "{:,.4f}"

    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 7. DISTRIBUTE MODULE TYPES
# ============================================================

st.header("Module Type Distribution")


# ============================================================
# SORT LARGEST MODULE TYPES FIRST
# ============================================================

working_df = (
    df
    .sort_values(
        "Calculated DC (MW)",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# CURRENT DC PER CLUSTER
# ============================================================

current_dc = np.zeros(
    cluster_count,
    dtype=float
)


assigned_cluster = []


# ============================================================
# DISTRIBUTION
# ============================================================

for _, row in working_df.iterrows():

    module_dc = float(
        row["Calculated DC (MW)"]
    )


    # --------------------------------------------------------
    # Difference between target and current
    # --------------------------------------------------------

    difference = (
        target_dc
        -
        current_dc
    )


    # --------------------------------------------------------
    # Prefer cluster that is furthest below target
    # --------------------------------------------------------

    below_target = np.where(
        difference > 0
    )[0]


    if len(below_target) > 0:

        selected = int(
            below_target[
                np.argmax(
                    difference[
                        below_target
                    ]
                )
            ]
        )

    else:

        selected = int(
            np.argmin(
                current_dc
            )
        )


    # --------------------------------------------------------
    # Assign COMPLETE module type
    # --------------------------------------------------------

    assigned_cluster.append(
        cluster_names[
            selected
        ]
    )


    current_dc[selected] += (
        module_dc
    )


# ============================================================
# ADD CLUSTER
# ============================================================

working_df["Cluster"] = (
    assigned_cluster
)


# ============================================================
# SORT CLUSTERS
# ============================================================

cluster_order = {
    name: index
    for index, name
    in enumerate(cluster_names)
}


working_df["_order"] = (
    working_df["Cluster"]
    .map(cluster_order)
)


result_df = (
    working_df
    .sort_values(
        [
            "_order",
            "Calculated DC (MW)"
        ],
        ascending=[
            True,
            False
        ]
    )
    .drop(
        columns="_order"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# RESULT COLUMN ORDER
# ============================================================

result_columns = [
    "Cluster",
    "Module Type",
    "(Wp)",
    "Std PV Eff(%)",
    "No of Modules",
    "Area of 1 module",
    "Total area(m2)",
    "Calculated DC (MW)"
]


result_columns = [
    column
    for column in result_columns
    if column in result_df.columns
]


result_df = result_df[
    result_columns
]


# ============================================================
# SHOW RESULT
# ============================================================

st.subheader(
    "Final Module Distribution"
)

st.dataframe(
    result_df.style.format({

        "(Wp)": "{:,.2f}",

        "Std PV Eff(%)": "{:,.2f}",

        "No of Modules": "{:,.0f}",

        "Area of 1 module":
            "{:,.4f}",

        "Total area(m2)":
            "{:,.2f}",

        "Calculated DC (MW)":
            "{:,.6f}"

    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

st.header("Final Cluster Summary")


actual_dc = (
    result_df
    .groupby(
        "Cluster"
    )["Calculated DC (MW)"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0
    )
    .to_numpy()
)


module_type_count = (
    result_df
    .groupby(
        "Cluster"
    )["Module Type"]
    .count()
    .reindex(
        cluster_names,
        fill_value=0
    )
    .to_numpy()
)


module_count = (
    result_df
    .groupby(
        "Cluster"
    )["No of Modules"]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0
    )
    .to_numpy()
)


area = (
    result_df
    .groupby(
        "Cluster"
    )["Total area(m2)"
    ]
    .sum()
    .reindex(
        cluster_names,
        fill_value=0
    )
    .to_numpy()
)


final_summary = pd.DataFrame({

    "Cluster":
        cluster_names,

    "Inverters":
        inverter_distribution,

    "Inverter Share (%)":
        inverter_proportion * 100,

    "AC":
        cluster_ac,

    "Module Types":
        module_type_count,

    "No of Modules":
        module_count,

    "Total Area (m2)":
        area,

    "Target DC (MW)":
        target_dc,

    "Actual DC (MW)":
        actual_dc,

    "DC Difference (MW)":
        actual_dc - target_dc
})


st.dataframe(
    final_summary.style.format({

        "Inverter Share (%)":
            "{:.2f}%",

        "AC":
            "{:,.2f}",

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
# 9. VALIDATION
# ============================================================

st.header("Validation")


original_dc = float(
    df["Calculated DC (MW)"].sum()
)


distributed_dc = float(
    result_df["Calculated DC (MW)"].sum()
)


original_modules = int(
    df["No of Modules"].sum()
)


distributed_modules = int(
    result_df["No of Modules"].sum()
)


original_types = int(
    df["Module Type"].nunique()
)


distributed_types = int(
    result_df["Module Type"].nunique()
)


has_duplicates = (
    result_df["Module Type"]
    .duplicated()
    .any()
)


validation = st.columns(4)


with validation[0]:

    if np.isclose(
        original_dc,
        distributed_dc,
        atol=1e-9
    ):

        st.success(
            "✓ DC Preserved"
        )

    else:

        st.error(
            "✗ DC Mismatch"
        )


with validation[1]:

    if (
        original_modules
        ==
        distributed_modules
    ):

        st.success(
            "✓ Modules Preserved"
        )

    else:

        st.error(
            "✗ Module Mismatch"
        )


with validation[2]:

    if (
        original_types
        ==
        distributed_types
    ):

        st.success(
            "✓ Module Types Preserved"
        )

    else:

        st.error(
            "✗ Module Type Mismatch"
        )


with validation[3]:

    if not has_duplicates:

        st.success(
            "✓ No Duplicates"
        )

    else:

        st.error(
            "✗ Duplicate Module Type"
        )


# ============================================================
# 10. DOWNLOAD
# ============================================================

st.header("Download")


excel_buffer = io.BytesIO()


with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl"
) as writer:

    result_df.to_excel(
        writer,
        sheet_name="Module Distribution",
        index=False
    )

    final_summary.to_excel(
        writer,
        sheet_name="Cluster Summary",
        index=False
    )

    df.to_excel(
        writer,
        sheet_name="Input Data",
        index=False
    )


st.download_button(
    "⬇️ Download Excel",
    data=excel_buffer.getvalue(),
    file_name="Solar_Cluster_Distribution.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True
)


# ============================================================
# CSV
# ============================================================

csv_data = result_df.to_csv(
    index=False
)


st.download_button(
    "⬇️ Download CSV",
    data=csv_data,
    file_name="Solar_Module_Distribution.csv",
    mime="text/csv",
    use_container_width=True
)
