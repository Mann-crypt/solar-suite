
from pathlib import Path
import html
import json
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

PAGE_SIZE = 20

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    .main-header {
        padding: 23px 26px;
        border-radius: 16px;
        margin-bottom: 22px;
        background: linear-gradient(115deg, #123c69 0%, #176b87 55%, #20a39e 100%);
        color: white;
        box-shadow: 0 5px 18px rgba(20, 80, 110, 0.16);
    }

    .main-header h1 {
        color: white !important;
        font-size: 29px;
        font-weight: 750;
        margin: 0;
        padding: 0;
    }

    .main-header p {
        color: #e4f5fb;
        margin: 7px 0 0 0;
        font-size: 14px;
    }

    .overview-card {
        padding: 16px 18px;
        border-radius: 13px;
        background: linear-gradient(135deg, #edf7ff, #e3f5f2);
        border: 1px solid #d2e8ed;
        margin-bottom: 8px;
    }

    .overview-label {
        color: #426175;
        font-size: 12px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .overview-value {
        color: #123c69;
        font-size: 25px;
        font-weight: 750;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 750;
        margin: 15px 0 8px 0;
    }

    .location-tile {
        border: 1px solid #dbe5ed;
        border-radius: 9px;
        padding: 7px 10px;
        background: rgba(125, 160, 185, 0.08);
        margin: 4px 0 8px 0;
        min-height: 49px;
    }

    .pss-info-label {
        color: #718096;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.45px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .pss-info-value {
        color: var(--text-color);
        font-size: 13px;
        font-weight: 650;
        overflow-wrap: anywhere;
    }

    .card-subtitle {
        font-size: 12px;
        opacity: 0.8;
        margin-top: 2px;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
        border-color: rgba(128, 145, 160, 0.35);
    }

    div[data-testid="stExpander"] details summary {
        font-size: 13px;
        font-weight: 650;
    }

    div[data-testid="stButton"] button {
        border-radius: 9px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        font-weight: 650;
    }

    @media (prefers-color-scheme: dark) {
        .overview-card {
            background: linear-gradient(135deg, #192d40, #193b3b);
            border-color: #35505e;
        }

        .overview-label {
            color: #c4d7e2;
        }

        .overview-value {
            color: #d7f1ff;
        }

        .location-tile {
            border-color: #3a4c5b;
            background: rgba(200, 220, 235, 0.05);
        }

        .pss-info-label {
            color: #b0c0cd;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """Convert a cell value to clean display text."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_column_name(value):
    """Normalise column names for flexible matching."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def find_column(columns, exact_names=(), contains_names=()):
    """Find a column using preferred exact names, then partial matches."""
    normalised = {
        column: normalise_column_name(column)
        for column in columns
    }

    for expected in exact_names:
        expected_normalised = normalise_column_name(expected)

        for column, actual in normalised.items():
            if actual == expected_normalised:
                return column

    for expected in contains_names:
        expected_normalised = normalise_column_name(expected)

        for column, actual in normalised.items():
            if expected_normalised in actual:
                return column

    return None


def first_nonempty(*values):
    """Return the first nonempty value."""
    for value in values:
        text_value = clean_text(value)

        if text_value:
            return text_value

    return ""


def format_capacity(value):
    """Format capacity values for display."""
    if pd.isna(value):
        return "—"

    try:
        number = float(value)

        if number.is_integer():
            return f"{int(number):,} MW"

        return f"{number:,.2f} MW"

    except (ValueError, TypeError):
        text_value = clean_text(value)

        if not text_value:
            return "—"

        return text_value


def display_value(value):
    """Return a display-safe value."""
    text_value = clean_text(value)
    return text_value if text_value else "—"


def render_location_tile(location):
    """
    Render the location tile as actual HTML.

    Important:
    HTML is rendered with st.markdown(..., unsafe_allow_html=True),
    so it will not appear as raw HTML code on the page.
    """
    safe_location = html.escape(display_value(location))

    st.markdown(
        """
        <div class="pss-info location-tile">
            <div class="pss-info-label">📍 Location</div>
            <div class="pss-info-value">LOCATION_PLACEHOLDER</div>
        </div>
        """.replace("LOCATION_PLACEHOLDER", safe_location),
        unsafe_allow_html=True,
    )


def render_copy_button(pss_name, unique_id):
    """Show a small isolated copy button without expanding the card."""
    safe_name = json.dumps(str(pss_name))

    copy_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                background: transparent;
            }

            button {
                width: 100%;
                height: 34px;
                border: 1px solid #b9cbd7;
                border-radius: 9px;
                background: transparent;
                color: inherit;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
            }

            button:hover {
                background: rgba(100, 160, 190, 0.14);
                border-color: #4299b5;
            }

            #status {
                font-size: 10px;
                text-align: center;
                margin-top: 2px;
            }
        </style>
    </head>
    <body>
        <button id="copyButton" type="button" title="Copy PSS name">
            ⚡
        </button>
        <div id="status"></div>

        <script>
            const pssName = PSS_NAME;
            const button = document.getElementById("copyButton");
            const status = document.getElementById("status");

            button.addEventListener("click", async function () {
                try {
                    await navigator.clipboard.writeText(pssName);
                    status.textContent = "Copied!";
                } catch (error) {
                    const area = document.createElement("textarea");
                    area.value = pssName;
                    area.style.position = "fixed";
                    area.style.left = "-10000px";
                    document.body.appendChild(area);
                    area.select();

                    const copied = document.execCommand("copy");
                    area.remove();

                    status.textContent = copied ? "Copied!" : "Copy failed";
                }

                setTimeout(function () {
                    status.textContent = "";
                }, 1500);
            });
        </script>
    </body>
    </html>
    """.replace("PSS_NAME", safe_name)

    components.html(
        copy_html,
        height=48,
        scrolling=False,
    )


# ============================================================
# LOAD EXCEL
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel(file_path, modified_time):
    """Load Excel data and use the first worksheet."""
    return pd.read_excel(file_path)


if not EXCEL_PATH.exists():
    st.error(f"Excel file not found: {EXCEL_PATH}")
    st.info(
        "Please keep intimation.xlsx in the main project folder, "
        "one level above the pages folder."
    )
    st.stop()

try:
    df = load_excel(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )

except Exception as error:
    st.error("Unable to read intimation.xlsx.")
    st.exception(error)
    st.stop()


if df.empty:
    st.warning("The Excel file contains no records.")
    st.stop()


# ============================================================
# FLEXIBLE COLUMN MAPPING
# ============================================================

columns = list(df.columns)

COL_PSS = find_column(
    columns,
    exact_names=("PSS", "PSS Name", "Name of PSS", "PSS_Name"),
    contains_names=("pss name",),
)

COL_STATE = find_column(
    columns,
    exact_names=("State", "Location", "State/Location"),
    contains_names=("state", "location"),
)

COL_CAPACITY = find_column(
    columns,
    exact_names=(
        "Capacity (MW)",
        "Capacity MW",
        "Capacity",
        "Installed Capacity",
    ),
    contains_names=("capacity",),
)

COL_PRIORITY = find_column(
    columns,
    exact_names=("Priority as per 003 sheet",),
)

if COL_PRIORITY is None:
    COL_PRIORITY = find_column(
        columns,
        exact_names=("Priority as per Amresh Sir 28.07.2023",),
    )

if COL_PRIORITY is None:
    COL_PRIORITY = next(
        (
            column
            for column in columns
            if "priority" in normalise_column_name(column)
        ),
        None,
    )

COL_POC1 = find_column(
    columns,
    exact_names=("POC1", "POC 1", "POC-1"),
    contains_names=("poc1", "poc 1", "poc-1"),
)

COL_POC2 = find_column(
    columns,
    exact_names=("POC2", "POC 2", "POC-2"),
    contains_names=("poc2", "poc 2", "poc-2"),
)

COL_POC3 = find_column(
    columns,
    exact_names=("POC3", "POC 3", "POC-3"),
    contains_names=("poc3", "poc 3", "poc-3"),
)

COL_LAT = find_column(
    columns,
    exact_names=("Latitude", "Lat"),
    contains_names=("latitude",),
)

COL_LONG = find_column(
    columns,
    exact_names=("Longitude", "Long", "Lon", "Lng"),
    contains_names=("longitude",),
)

COL_CHECK = find_column(
    columns,
    exact_names=("Check Status", "Check_Status", "Status"),
    contains_names=("check status",),
)


if COL_PSS is None:
    st.error(
        "PSS column could not be detected. "
        "Please check the column names in intimation.xlsx."
    )
    st.write("Available Excel columns:", columns)
    st.stop()


# ============================================================
# STANDARDISE VALUES USED BY FILTERS
# ============================================================

working_df = df.copy()

working_df["_display_pss"] = working_df[COL_PSS].map(clean_text)

if COL_STATE is not None:
    working_df["_display_state"] = working_df[COL_STATE].map(clean_text)
else:
    working_df["_display_state"] = ""

if COL_POC1 is not None:
    working_df["_display_poc1"] = working_df[COL_POC1].map(clean_text)
else:
    working_df["_display_poc1"] = ""

working_df = working_df[
    working_df["_display_pss"] != ""
].copy()

if working_df.empty:
    st.warning("No records with a valid PSS name were found.")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>📋 Intimation Dashboard</h1>
        <p>Power System Sites · Intimation Records · Quick Overview</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS
# ============================================================

st.markdown(
    '<div class="section-heading">🔎 Filters</div>',
    unsafe_allow_html=True,
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

pss_values = sorted(
    working_df["_display_pss"].dropna().unique().tolist(),
    key=str.casefold,
)

with filter_col1:
    selected_pss = st.selectbox(
        "PSS Name",
        options=["All PSS"] + pss_values,
        index=0,
        key="intimation_pss_filter",
        help="Type to search for a PSS.",
    )

filtered_df = working_df.copy()

if selected_pss != "All PSS":
    filtered_df = filtered_df[
        filtered_df["_display_pss"] == selected_pss
    ]

poc1_values = sorted(
    filtered_df["_display_poc1"].dropna().unique().tolist(),
    key=str.casefold,
)

poc1_values = [value for value in poc1_values if value]

with filter_col2:
    selected_poc1 = st.selectbox(
        "POC1",
        options=["All POC1"] + poc1_values,
        index=0,
        key="intimation_poc1_filter",
    )

if selected_poc1 != "All POC1":
    filtered_df = filtered_df[
        filtered_df["_display_poc1"] == selected_poc1
    ]

state_values = sorted(
    filtered_df["_display_state"].dropna().unique().tolist(),
    key=str.casefold,
)

state_values = [value for value in state_values if value]

with filter_col3:
    selected_state = st.selectbox(
        "State / Location",
        options=["All States"] + state_values,
        index=0,
        key="intimation_state_filter",
    )

if selected_state != "All States":
    filtered_df = filtered_df[
        filtered_df["_display_state"] == selected_state
    ]


# ============================================================
# OVERVIEW METRICS
# ============================================================

matching_records = len(filtered_df)

if COL_CAPACITY is not None:
    capacity_values = pd.to_numeric(
        filtered_df[COL_CAPACITY],
        errors="coerce",
    )
    total_capacity = capacity_values.sum()
    capacity_text = f"{total_capacity:,.2f} MW"
else:
    capacity_text = "N/A"

metric1, metric2 = st.columns(2)

with metric1:
    st.markdown(
        f"""
        <div class="overview-card">
            <div class="overview-label">Matching Records</div>
            <div class="overview-value">{matching_records:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f"""
        <div class="overview-card">
            <div class="overview-label">Total Capacity</div>
            <div class="overview-value">{html.escape(capacity_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGINATION
# ============================================================

st.markdown(
    '<div class="section-heading">📑 PSS Records</div>',
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.info("No records match the selected filters.")
    st.stop()

total_pages = max(1, math.ceil(len(filtered_df) / PAGE_SIZE))

if "intimation_page" not in st.session_state:
    st.session_state["intimation_page"] = 1

current_page = st.session_state["intimation_page"]

if current_page > total_pages:
    current_page = total_pages

if current_page < 1:
    current_page = 1

st.session_state["intimation_page"] = current_page

page_col1, page_col2, page_col3 = st.columns([1, 2, 1])

with page_col1:
    if st.button(
        "← Previous",
        disabled=current_page <= 1,
        use_container_width=True,
        key="intimation_previous_page",
    ):
        st.session_state["intimation_page"] = current_page - 1
        st.rerun()

with page_col2:
    st.markdown(
        f"<div style='text-align:center;padding-top:8px;'>"
        f"Page <b>{current_page}</b> of <b>{total_pages}</b>"
        f" &nbsp;·&nbsp; {len(filtered_df):,} records"
        f"</div>",
        unsafe_allow_html=True,
    )

with page_col3:
    if st.button(
        "Next →",
        disabled=current_page >= total_pages,
        use_container_width=True,
        key="intimation_next_page",
    ):
        st.session_state["intimation_page"] = current_page + 1
        st.rerun()


start_index = (current_page - 1) * PAGE_SIZE
end_index = start_index + PAGE_SIZE

page_df = filtered_df.iloc[start_index:end_index]


# ============================================================
# CARD RENDERING
# ============================================================

def render_pss_card(row, card_number):
    """Render one compact PSS card with expandable details."""

    pss_name = clean_text(row.get("_display_pss", ""))
    state_name = clean_text(row.get("_display_state", ""))

    capacity = (
        format_capacity(row[COL_CAPACITY])
        if COL_CAPACITY is not None
        else "—"
    )

    poc1 = (
        display_value(row[COL_POC1])
        if COL_POC1 is not None
        else "—"
    )

    safe_pss_name = html.escape(display_value(pss_name))
    safe_capacity = html.escape(capacity)
    safe_poc1 = html.escape(poc1)

    with st.container(border=True):

        header_col, copy_col = st.columns([5, 1])

        with header_col:
            st.markdown(
                f"""
                <div style="font-size:16px;font-weight:750;
                            line-height:1.35;overflow-wrap:anywhere;">
                    {safe_pss_name}
                </div>
                <div class="card-subtitle">
                    ⚡ PSS Information
                </div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            render_copy_button(
                pss_name,
                f"copy_{card_number}",
            )

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(
                f"""
                <div class="pss-info location-tile">
                    <div class="pss-info-label">⚡ Capacity</div>
                    <div class="pss-info-value">{safe_capacity}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with info_col2:
            # FIX: Render HTML instead of displaying the markup as text.
            render_location_tile(state_name)

        st.markdown(
            f"""
            <div class="pss-info-label">👤 POC1</div>
            <div class="pss-info-value">{safe_poc1}</div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View details", expanded=False):

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**⭐ Priority**")

                if COL_PRIORITY is not None:
                    st.write(display_value(row[COL_PRIORITY]))
                else:
                    st.write("—")

                st.markdown("**👤 POC2**")

                if COL_POC2 is not None:
                    st.write(display_value(row[COL_POC2]))
                else:
                    st.write("—")

                st.markdown("**👤 POC3**")

                if COL_POC3 is not None:
                    st.write(display_value(row[COL_POC3]))
                else:
                    st.write("—")

            with detail_col2:
                st.markdown("**🌐 Latitude**")

                if COL_LAT is not None:
                    st.write(display_value(row[COL_LAT]))
                else:
                    st.write("—")

                st.markdown("**🌐 Longitude**")

                if COL_LONG is not None:
                    st.write(display_value(row[COL_LONG]))
                else:
                    st.write("—")

                st.markdown("**✅ Check Status**")

                if COL_CHECK is not None:
                    st.write(display_value(row[COL_CHECK]))
                else:
                    st.write("—")


# ============================================================
# TWO-COLUMN CARD LAYOUT
# ============================================================

left_column, right_column = st.columns(2, gap="medium")

for position, (_, record) in enumerate(page_df.iterrows()):

    absolute_card_number = start_index + position

    if position % 2 == 0:
        with left_column:
            render_pss_card(record, absolute_card_number)
    else:
        with right_column:
            render_pss_card(record, absolute_card_number)


# ============================================================
# BOTTOM PAGINATION
# ============================================================

st.divider()

bottom_col1, bottom_col2, bottom_col3 = st.columns([1, 2, 1])

with bottom_col1:
    if st.button(
        "← Previous page",
        disabled=current_page <= 1,
        use_container_width=True,
        key="intimation_bottom_previous",
    ):
        st.session_state["intimation_page"] = current_page - 1
        st.rerun()

with bottom_col2:
    st.markdown(
        f"<div style='text-align:center;padding-top:8px;'>"
        f"Showing {start_index + 1}–{min(end_index, len(filtered_df))} "
        f"of {len(filtered_df):,} matching records"
        f"</div>",
        unsafe_allow_html=True,
    )

with bottom_col3:
    if st.button(
        "Next page →",
        disabled=current_page >= total_pages,
        use_container_width=True,
        key="intimation_bottom_next",
    ):
        st.session_state["intimation_page"] = current_page + 1
        st.rerun()
