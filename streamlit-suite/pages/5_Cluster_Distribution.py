# ============================================================
# SOLAR MODULE -> INVERTER CLUSTER DISTRIBUTION
# Lightweight / Stable Streamlit Version
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

st.title("☀️ Solar Module Cluster Distribution")

st.write(
    "Assign complete module types to inverter clusters according "
    "to inverter proportion. Module types are never split or "
    "duplicated between clusters."
)


# ============================================================
# STEP 1
# ============================================================

st.header("Step 1: Cluster Information")

cluster_count = st.number_input(
    "Enter number of clusters",
    min_value=1,
    step=1,
    value=None,
    placeholder="Enter cluster count"
)


if cluster_count is None:
    st.info("Enter the number of clusters to continue.")
    st.stop()


cluster_count = int(cluster_count)


# ============================================================
# STEP 2
# ============================================================

st.header("Step 2: Inverter Distribution")

st.write(
    "Enter the number of inverters belonging to each cluster."
)

inverter_distribution = []

valid_inverter_input = True

for i in range(cluster_count):

    inverter_value = st.number_input(
        f"Cluster {i + 1} - Number of Inverters",
        min_value=1,
        step=1,
        value=None,
        placeholder="Enter inverter count",
        key=f"inverter_count_{i}"
    )

    if inverter_value is None:
        valid_inverter_input = False
        inverter_distribution.append(0)
    else:
        inverter_distribution.append(
            int(inverter_value)
        )


if not valid_inverter_input:

    st.info(
        "Enter inverter count for every cluster to continue."
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
# INVERTER PROPORTIONS
# ============================================================

cluster_names = [
    f"Cluster {i + 1}"
    for i in range(cluster_count)
]


cluster_proportions = (
    np.array(
        inverter_distribution,
        dtype=float
    )
    /
    total_inverters
)


# ============================================================
# STEP 3
# ============================================================

st.header("Step 3: Enter Total AC")

st.write(
    "DC does not need to be entered. DC will be calculated "
    "automatically from the module data."
)

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
    total_ac *
    cluster_proportions
)


# ============================================================
# SHOW INVERTER + AC DISTRIBUTION
# ============================================================

ac_summary = pd.DataFrame({

    "Cluster": cluster_names,

    "Inverters": inverter_distribution,

    "Inverter Proportion (%)":
        cluster_proportions * 100,

    "AC":
        cluster_ac

})


st.subheader("Inverter / AC Distribution")

st.dataframe(
    ac_summary.style.format({

        "Inverter Proportion (%)":
            "{:.2f}%",

        "AC":
            "{:,.2f}"

    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STEP 4
# ============================================================

st.header("Step 4: Module Data")

input_method = st.radio(
    "How do you want to provide module data?",
    [
        "Upload File",
        "Manual Entry"
    ],
    index=None
)


if input_method is None:

    st.info(
        "Select Upload File or Manual Entry."
    )

    st.stop()


# ============================================================
# OPTION A: UPLOAD FILE
# ============================================================

if input_method == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload module data",
        type=["xlsx", "xls", "csv"],
        help=(
            "Required columns: Module Type, (Wp), "
            "No of Modules, Area of 1 module"
        )
    )


    if uploaded_file is None:

        st.info(
            "Upload your Excel or CSV file to continue."
        )

        st.stop()


    try:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            f"Unable to read the uploaded file: {e}"
        )

        st.stop()


# ============================================================
# OPTION B: MANUAL ENTRY
# ============================================================

else:

    st.write(
        "Enter one row for each module type."
    )

    st.caption(
        "Do not enter the same module type more than once."
    )


    manual_row_count = st.number_input(
        "Enter number of module types",
        min_value=1,
        step=1,
        value=None,
        placeholder="Enter number of module types"
    )


    if manual_row_count is None:

        st.info(
            "Enter the number of module types."
        )

        st.stop()


    manual_row_count = int(
        manual_row_count
    )


    manual_rows = []


    for i in range(manual_row_count):

        st.markdown(
            f"**Module Type {i + 1}**"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            module_type = st.text_input(
                "Module Type",
                key=f"module_type_{i}",
                placeholder="Enter module type"
            )


        with col2:

            wp = st.number_input(
                "Wp",
                min_value=0.0,
                step=0.01,
                value=None,
                placeholder="Enter Wp",
                key=f"wp_{i}"
            )


        with col3:

            efficiency = st.number_input(
                "Std PV Eff (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                value=None,
                placeholder="Enter efficiency",
                key=f"efficiency_{i}"
            )


        col4, col5 = st.columns(2)


        with col4:

            module_count = st.number_input(
                "No of Modules",
                min_value=0,
                step=1,
                value=None,
                placeholder="Enter module count",
                key=f"module_count_{i}"
            )


        with col5:

            area = st.number_input(
                "Area of 1 module",
                min_value=0.0,
                step=0.0001,
                value=None,
                placeholder="Enter module area",
                key=f"area_{i}"
            )


        manual_rows.append({

            "Module Type": module_type,

            "(Wp)": wp,

            "Std PV Eff(%)": efficiency,

            "No of Modules": module_count,

            "Area of 1 module": area

        })


    df = pd.DataFrame(
        manual_rows
    )


# ============================================================
# STEP 5
# ============================================================

st.header("Step 5: Validate Module Data")


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
        "The following required columns are missing:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

df = df.dropna(
    how="all"
).reset_index(
    drop=True
)


# Module type

df["Module Type"] = (
    df["Module Type"]
    .astype(str)
    .str.strip()
)


# Remove empty module types

df = df[
    (
        df["Module Type"] != ""
    )
    &
    (
        df["Module Type"].str.lower()
        != "nan"
    )
].copy()


if df.empty:

    st.error(
        "No valid module data was found."
    )

    st.stop()


# ============================================================
# NUMERIC COLUMNS
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


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CHECK INVALID VALUES
# ============================================================

if df[
    "(Wp)"
].isna().any():

    st.error(
        "Invalid or missing Wp value found."
    )

    st.stop()


if df[
    "No of Modules"
].isna().any():

    st.error(
        "Invalid or missing No of Modules value found."
    )

    st.stop()


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


# ============================================================
# CHECK DUPLICATE MODULE TYPES
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
        .unique()
        .tolist()
    )


    st.error(
        "Duplicate Module Type detected."
    )

    st.write(
        "The following module types appear more than once:"
    )

    st.write(
        duplicate_types
    )

    st.warning(
        "Each Module Type must appear only once. "
        "Please correct the input data."
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
# TOTAL PLANT DC
# ============================================================

total_dc = (
    df["Calculated DC (MW)"]
    .sum()
)


# ============================================================
# PLANT SUMMARY
# ============================================================

st.subheader("Plant Summary")

metric1, metric2, metric3 = st.columns(3)


with metric1:

    st.metric(
        "Total AC",
        f"{total_ac:,.2f}"
    )


with metric2:

    st.metric(
        "Calculated DC",
        f"{total_dc:,.4f} MW"
    )


with metric3:

    st.metric(
        "Module Types",
        f"{len(df):,}"
    )


# ============================================================
# MODULE DATA PREVIEW
# ============================================================

with st.expander(
    "View Module Data + Calculated DC"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STEP 6
# ============================================================

st.header("Step 6: Calculate Cluster DC Targets")


target_dc = (
    total_dc
    *
    cluster_proportions
)


target_summary = pd.DataFrame({

    "Cluster": cluster_names,

    "Inverters": inverter_distribution,

    "Inverter Share (%)":
        cluster_proportions * 100,

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
# STEP 7
# ============================================================

st.header("Step 7: Assign Module Types to Clusters")


st.info(
    "Each Module Type is assigned completely to ONE cluster. "
    "No Module Type is split or duplicated."
)


# ============================================================
# DISTRIBUTION ALGORITHM
# ============================================================
#
# Important:
#
# Module types are indivisible.
#
# Example:
#
# Type A = 50 MW
#
# It must go entirely to one cluster.
#
# The algorithm tries to keep cluster DC as close as possible
# to the inverter-proportional target.
#
# ============================================================


module_df = (
    df
    .sort_values(
        "Calculated DC (MW)",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


current_dc = np.zeros(
    cluster_count,
   
