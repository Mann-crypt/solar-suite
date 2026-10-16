
# pages/0_Intimation.py

from pathlib import Path
import html
import json
import math
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# THEME-AWARE CSS — LIGHT MODE + DARK MODE
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: light dark;
    }

    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--background-color);
    }

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    .dashboard-header {
        background: linear-gradient(120deg, #1746a2 0%, #2478d4 55%, #16a6a1 100%);
        padding: 25px 28px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 8px 25px rgba(30, 90, 170, 0.18);
    }

    .dashboard-header h1 {
        color: #ffffff !important;
        font-size: clamp(1.65rem, 3vw, 2.25rem);
        line-height: 1.2;
        margin: 0;
        font-weight: 800;
    }

    .dashboard-header p {
        color: #e9f4ff !important;
        margin: 8px 0 0 0;
        font-size: 0.95rem;
    }

    .header-label {
        color: #d9f5ff !important;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .section-heading {
        color: var(--text-color) !important;
        font-size: 1.05rem;
        font-weight: 750;
        margin: 8px 0 12px 0;
    }

    .filter-caption {
        color: var(--text-color) !important;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .metric-card {
        padding: 18px 20px;
        border-radius: 15px;
        border: 1px solid var(--border-color, rgba(128,128,128,0.25));
        background: var(--secondary-background-color);
        min-height: 105px;
    }

    .metric-label {
        color: var(--text-color) !important;
        opacity: 0.75;
        font-size: 0.84rem;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--text-color) !important;
        font-size: clamp(1.45rem, 2.5vw, 2rem);
        font-weight: 800;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .metric-records {
        border-left: 5px solid #4385f5;
    }

    .metric-capacity {
        border-left: 5px solid #13a88a;
    }

    .pss-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color, rgba(128,128,128,0.25));
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 7px;
        min-height: 145px;
    }

    .pss-title-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 13px;
    }

    .pss-icon {
        width: 38px;
        height: 38px;
        min-width: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        background: rgba(65, 133, 244, 0.13);
        font-size: 20px;
    }

    .pss-heading {
        color: var(--text-color) !important;
        font-size: 0.96rem;
        font-weight: 800;
        line-height: 1.35;
        overflow-wrap: anywhere;
        margin: 0;
    }

    .pss-subheading {
        color: var(--text-color) !important;
        opacity: 0.7;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    .pss-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 9px;
    }

    .pss-info {
        background: var(--background-color);
        border: 1px solid var(--border-color, rgba(128,128,128,0.18));
        border-radius: 10px;
        padding: 10px;
        min-width: 0;
    }

    .pss-info-label {
        color: var(--text-color) !important;
        opacity: 0.68;
        font-size: 0.72rem;
        margin-bottom: 5px;
    }

    .pss-info-value {
        color: var(--text-color) !important;
        font-size: 0.88rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .capacity-tile {
        border-left: 3px solid #13a88a;
    }

    .state-tile {
        border-left: 3px solid #4385f5;
    }

    .detail-tile {
        background: var(--background-color);
        border: 1px solid var(--border-color, rgba(128,128,128,0.2));
        border-radius: 10px;
        padding: 11px;
        margin-bottom: 9px;
        min-height: 68px;
    }

    .detail-label {
        color: var(--text-color) !important;
        opacity: 0.7;
        font-size: 0.74rem;
        margin-bottom: 6px;
    }

    .detail-value {
        color: var(--text-color) !important;
        font-size: 0.9rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 0.78rem;
        font-weight: 750;
        background: rgba(19, 168, 138, 0.14);
        color: #13a88a !important;
        overflow-wrap: anywhere;
    }

    .results-caption {
        color: var(--text-color) !important;
        opacity: 0.7;
        font-size: 0.84rem;
        margin: 2px 0 12px 0;
    }

    .empty-state {
        background: var(--secondary-background-color);
        color: var(--text-color) !important;
        border: 1px dashed var(--border-color, rgba(128,128,128,0.35));
        border-radius: 15px;
        padding: 30px 20px;
        text-align: center;
    }

    .empty-state h3 {
        color: var(--text-color) !important;
        margin-bottom: 8px;
    }

    .empty-state p {
        color: var(--text-color) !important;
        opacity: 0.75;
    }

    .page-label {
        color: var(--text-color) !important;
        opacity: 0.8;
        text-align: center;
        padding-top: 7px;
        font-size: 0.86rem;
    }

    div[data-testid="stExpander"] {
        border-color: var(--border-color, rgba(128,128,128,0.25));
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] details summary {
        font-weight: 650;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        font-weight: 650;
    }

    div[data-testid="stButton"] > button {
        border-radius: 9px;
        font-weight: 650;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .dashboard-header {
            padding: 20px;
        }

        .metric-card {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_name(value):
    """Normalize a column name for flexible matching."""
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def find_column(columns, aliases):
    """
    Find a column using normalized aliases.
    Exact normalized matches are checked before partial matches.
    """
    normalized_columns = {
        normalize_name(column): column for column in columns
    }

    normalized_aliases = [normalize_name(alias) for alias in aliases]

    for alias in normalized_aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    for alias in normalized_aliases:
        if not alias:
            continue

        for normalized_column, original_column in normalized_columns.items():
            if alias in normalized_column:
                return original_column

    return None


def display_value(value, default="—"):
    """Convert a cell to a clean display string."""
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:,.4f}".rstrip("0").rstrip(".")

    return str(value).strip() or default


def unique_values(series):
    """Return clean, unique, sorted dropdown values."""
    if series is None:
        return []

    values = []

    for value in series.dropna().tolist():
        cleaned = str(value).strip()

        if cleaned and cleaned.lower() not in {
            "nan", "none", "null", "nat"
        }:
            values.append(cleaned)

    return sorted(set(values), key=str.casefold)


def escape_html(value):
    return html.escape(display_value(value))


def render_metric(label, value, css_class):
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail(label, value, status=False):
    value_text = display_value(value)

    if status:
        value_html = (
            '<span class="status-pill">'
            + html.escape(value_text)
            + "</span>"
        )
    else:
        value_html = (
            '<div class="detail-value">'
            + html.escape(value_text)
            + "</div>"
        )

    st.markdown(
        f"""
        <div class="detail-tile">
            <div class="detail-label">{html.escape(label)}</div>
            {value_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(pss_name, unique_id):
    """
    Render a standalone copy button.
    It runs in its own iframe, so clicking it does not expand the card.
    """
    value_json = json.dumps(str(pss_name))

    button_html = """
    <div style="display:flex;justify-content:flex-end;margin:0 0 5px 0;">
      <button id="copy-btn"
        style="
          cursor:pointer;
          border:1px solid rgba(128,128,128,.35);
          border-radius:9px;
          padding:7px 10px;
          background:var(--secondary-background-color, #f5f5f5);
          color:var(--text-color, #222);
          font-size:13px;
          font-weight:650;
        ">
        ⚡ Copy PSS
      </button>
    </div>
    <script>
      const pssName = __PSS_VALUE__;

      document.getElementById("copy-btn").addEventListener("click", async () => {
        const button = document.getElementById("copy-btn");

        try {
          if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(pssName);
          } else {
            const textArea = document.createElement("textarea");
            textArea.value = pssName;
            textArea.style.position = "fixed";
            textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();

            const copied = document.execCommand("copy");
            textArea.remove();

            if (!copied) {
              throw new Error("Copy was not permitted");
            }
          }

          button.innerText = "✓ Copied!";
        } catch (error) {
          button.innerText = "Select PSS name manually";
        }
      });
    </script>
    """.replace("__PSS_VALUE__", value_json)

    components.html(
        button_html,
        height=43,
        scrolling=False,
    )


# ============================================================
# LOAD EXCEL
# ============================================================

possible_paths = [
    Path(__file__).resolve().parent.parent / "intimation.xlsx",
    Path.cwd() / "intimation.xlsx",
]

EXCEL_PATH = next(
    (path for path in possible_paths if path.exists()),
    None,
)

if EXCEL_PATH is None:
    st.error(
        "Excel file `intimation.xlsx` nahi mili. "
        "Please file ko project root folder mein rakho."
    )
    st.caption(
        "Expected location: project-folder/intimation.xlsx"
    )
    st.stop()

try:
    data = pd.read_excel(EXCEL_PATH)
except Exception as exc:
    st.error("Excel file read nahi ho paayi.")
    st.exception(exc)
    st.stop()

if data.empty:
    st.warning("Excel file mein koi records nahi hain.")
    st.stop()

data.columns = [str(column).strip() for column in data.columns]


# ============================================================
# FLEXIBLE COLUMN MAPPING
# ============================================================

pss_col = find_column(
    data.columns,
    [
        "PSS Name",
        "PSS",
        "Substation Name",
        "Pooling Substation",
        "Pooling Station",
    ],
)

capacity_col = find_column(
    data.columns,
    [
        "Capacity (MW)",
        "Capacity MW",
        "Capacity",
        "Installed Capacity",
        "Project Capacity (MW)",
    ],
)

state_col = find_column(
    data.columns,
    [
        "State",
        "Project State",
        "Location State",
    ],
)

poc1_col = find_column(
    data.columns,
    [
        "POC1",
        "POC 1",
        "POC-1",
        "POC 1 Name",
        "POC1 Name",
    ],
)

poc2_col = find_column(
    data.columns,
    [
        "POC 2",
        "POC2",
        "POC-2",
        "POC 2 Name",
        "POC2 Name",
    ],
)

poc3_col = find_column(
    data.columns,
    [
        "POC 3",
        "POC3",
        "POC-3",
        "POC 3 Name",
        "POC3 Name",
    ],
)

latitude_col = find_column(
    data.columns,
    [
        "Latitude",
        "Lat",
        "PSS Latitude",
        "Project Latitude",
    ],
)

longitude_col = find_column(
    data.columns,
    [
        "Longitude",
        "Long",
        "Lon",
        "Lng",
        "PSS Longitude",
        "Project Longitude",
    ],
)

check_status_col = find_column(
    data.columns,
    [
        "Check Status",
        "Status",
        "Checking Status",
        "Check",
    ],
)

# Priority column selection order matters.
priority_col = None

for preferred_priority in [
    "Priority as per 003 sheet",
    "Priority as per Amresh Sir 28.07.2023",
]:
    priority_col = find_column(
        data.columns,
        [preferred_priority],
    )

    if priority_col is not None:
        break

if priority_col is None:
    priority_col = next(
        (
            column
            for column in data.columns
            if "priority" in str(column).lower()
        ),
        None,
    )

if pss_col is None:
    st.error(
        "PSS Name column identify nahi ho paaya. "
        "Excel mein PSS Name ya PSS column hona chahiye."
    )
    st.write("Available columns:", list(data.columns))
    st.stop()


# ============================================================
# CLEAN DISPLAY DATA
# ============================================================

data = data.copy()
data["_original_index"] = range(len(data))

data[pss_col] = data[pss_col].apply(
    lambda value: (
        str(value).strip()
        if pd.notna(value) and str(value).strip()
        else ""
    )
)

data = data[data[pss_col] != ""].copy()

if data.empty:
    st.warning("PSS Name wale records nahi mile.")
    st.stop()


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-label">POWER • PROJECTS • TRACKING</div>
        <h1>⚡ Intimation Dashboard</h1>
        <p>Search, filter and review PSS intimation records.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS
# PSS → POC1 → State
# ============================================================

st.markdown(
    '<div class="section-heading">🔎 Search & Filters</div>',
    unsafe_allow_html=True,
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

pss_options = unique_values(data[pss_col])

with filter_col1:
    st.markdown(
        '<div class="filter-caption">PSS Name</div>',
        unsafe_allow_html=True,
    )

    selected_pss = st.selectbox(
        "PSS Name",
        options=["All PSS"] + pss_options,
        index=0,
        placeholder="Type or select a PSS name...",
        label_visibility="collapsed",
        key="intimation_pss_filter",
    )

# POC1 choices depend on selected PSS.
poc1_source = data.copy()

if selected_pss != "All PSS":
    poc1_source = poc1_source[
        poc1_source[pss_col] == selected_pss
    ]

poc1_options = (
    unique_values(poc1_source[poc1_col])
    if poc1_col is not None
    else []
)

with filter_col2:
    st.markdown(
        '<div class="filter-caption">POC 1</div>',
        unsafe_allow_html=True,
    )

    selected_poc1 = st.selectbox(
        "POC 1",
        options=["All POC 1"] + poc1_options,
        index=0,
        label_visibility="collapsed",
        key="intimation_poc1_filter",
        disabled=(poc1_col is None),
    )

# State choices depend on selected PSS + POC1.
state_source = poc1_source.copy()

if (
    poc1_col is not None
    and selected_poc1 != "All POC 1"
):
    state_source = state_source[
        state_source[poc1_col].apply(display_value)
        == selected_poc1
    ]

state_options = (
    unique_values(state_source[state_col])
    if state_col is not None
    else []
)

with filter_col3:
    st.markdown(
        '<div class="filter-caption">State</div>',
        unsafe_allow_html=True,
    )

    selected_state = st.selectbox(
        "State",
        options=["All States"] + state_options,
        index=0,
        label_visibility="collapsed",
        key="intimation_state_filter",
        disabled=(state_col is None),
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_data = data.copy()

if selected_pss != "All PSS":
    filtered_data = filtered_data[
        filtered_data[pss_col] == selected_pss
    ]

if (
    poc1_col is not None
    and selected_poc1 != "All POC 1"
):
    filtered_data = filtered_data[
        filtered_data[poc1_col].apply(display_value)
        == selected_poc1
    ]

if (
    state_col is not None
    and selected_state != "All States"
):
    filtered_data = filtered_data[
        filtered_data[state_col].apply(display_value)
        == selected_state
    ]

filtered_data = filtered_data.reset_index(drop=True)


# ============================================================
# SUMMARY METRICS
# ============================================================

st.markdown(
    '<div class="section-heading">📊 Overview</div>',
    unsafe_allow_html=True,
)

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    render_metric(
        "Matching Records",
        f"{len(filtered_data):,}",
        "metric-records",
    )

with metric_col2:
    total_capacity = 0.0

    if capacity_col is not None and not filtered_data.empty:
        capacity_values = pd.to_numeric(
            filtered_data[capacity_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("MW", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

        total_capacity = capacity_values.sum()

    render_metric(
        "Total Capacity",
        f"{total_capacity:,.2f} MW",
        "metric-capacity",
    )


# ============================================================
# RESULTS + PAGINATION
# ============================================================

st.markdown(
    '<div class="section-heading">📋 PSS Records</div>',
    unsafe_allow_html=True,
)

PAGE_SIZE = 20

total_records = len(filtered_data)
total_pages = max(1, math.ceil(total_records / PAGE_SIZE))

if "intimation_page" not in st.session_state:
    st.session_state.intimation_page = 1

# Keep page number valid after filters change.
st.session_state.intimation_page = min(
    max(1, st.session_state.intimation_page),
    total_pages,
)

if total_records == 0:
    st.session_state.intimation_page = 1

st.markdown(
    f"""
    <div class="results-caption">
        Showing {total_records:,} matching record(s)
        • {PAGE_SIZE} records per page
    </div>
    """,
    unsafe_allow_html=True,
)

if total_records == 0:
    st.markdown(
        """
        <div class="empty-state">
            <h3>🔍 No matching records</h3>
            <p>Try changing the PSS Name, POC 1 or State filter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# Pagination controls.
pagination_left, pagination_center, pagination_right = st.columns(
    [1, 2, 1]
)

with pagination_left:
    if st.button(
        "← Previous",
        disabled=(st.session_state.intimation_page <= 1),
        use_container_width=True,
        key="intimation_previous_page",
    ):
        st.session_state.intimation_page -= 1
        st.rerun()

with pagination_center:
    st.markdown(
        f"""
        <div class="page-label">
            Page {st.session_state.intimation_page} of {total_pages}
        </div>
        """,
        unsafe_allow_html=True,
    )

with pagination_right:
    if st.button(
        "Next →",
        disabled=(st.session_state.intimation_page >= total_pages),
        use_container_width=True,
        key="intimation_next_page",
    ):
        st.session_state.intimation_page += 1
        st.rerun()


start_index = (
    st.session_state.intimation_page - 1
) * PAGE_SIZE

end_index = min(start_index + PAGE_SIZE, total_records)

page_data = filtered_data.iloc[start_index:end_index]


# ============================================================
# CARD RENDERING
# Two columns, 20 records per page
# ============================================================

def render_pss_card(row, card_number):
    pss_name = display_value(row.get(pss_col, "—"))
    capacity = (
        display_value(row.get(capacity_col, "—"))
        if capacity_col is not None
        else "—"
    )
    state = (
        display_value(row.get(state_col, "—"))
        if state_col is not None
        else "—"
    )
    poc1 = (
        display_value(row.get(poc1_col, "—"))
        if poc1_col is not None
        else "—"
    )

    st.markdown(
        f"""
        <div class="pss-card">
            <div class="pss-title-row">
                <div class="pss-icon">⚡</div>
                <div style="min-width:0;">
                    <div class="pss-heading">
                        {escape_html(pss_name)}
                    </div>
                    <div class="pss-subheading">
                        PSS Intimation Record
                    </div>
                </div>
            </div>

            <div class="pss-info-grid">
                <div class="pss-info capacity-tile">
                    <div class="pss-info-label">Capacity</div>
                    <div class="pss-info-value">
                        {escape_html(capacity)}
                    </div>
                </div>

                <div class="pss-info state-tile">
                    <div class="pss-info-label">State</div>
                    <div class="pss-info-value">
                        {escape_html(state)}
                    </div>
                </div>

                <div class="pss-info">
                    <div class="pss-info-label">POC 1</div>
                    <div class="pss-info-value">
                        {escape_html(poc1)}
                    </div>
                </div>

                <div class="pss-info">
                    <div class="pss-info-label">Record</div>
                    <div class="pss-info-value">
                        {card_number}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_copy_button(
        pss_name,
        f"copy_{card_number}",
    )

    # Expanded details.
    with st.expander("View details", expanded=False):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            render_detail(
                "Priority",
                row.get(priority_col, "—")
                if priority_col is not None
                else "—",
            )

            render_detail(
                "POC 2",
                row.get(poc2_col, "—")
                if poc2_col is not None
                else "—",
            )

            render_detail(
                "POC 3",
                row.get(poc3_col, "—")
                if poc3_col is not None
                else "—",
            )

        with detail_col2:
            render_detail(
                "Latitude",
                row.get(latitude_col, "—")
                if latitude_col is not None
                else "—",
            )

            render_detail(
                "Longitude",
                row.get(longitude_col, "—")
                if longitude_col is not None
                else "—",
            )

            render_detail(
                "Check Status",
                row.get(check_status_col, "—")
                if check_status_col is not None
                else "—",
                status=True,
            )


left_column, right_column = st.columns(2, gap="medium")

for position, (_, row) in enumerate(page_data.iterrows()):
    card_number = start_index + position + 1

    if position % 2 == 0:
        with left_column:
            render_pss_card(row, card_number)
    else:
        with right_column:
            render_pss_card(row, card_number)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    f"Intimation Dashboard • {total_records:,} matching record(s) "
    f"• Displaying {start_index + 1}–{end_index}"
)
