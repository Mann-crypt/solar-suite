
import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM STYLING
# =========================================================
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --text-main: #172033;
        --text-muted: #64748b;
        --surface: #ffffff;
        --border: #e7edf5;
        --green: #059669;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top left, rgba(219,234,254,0.35), transparent 38%),
            #f5f7fb;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1600px !important;
    }

    /* Main header */
    .dashboard-hero {
        position: relative;
        overflow: hidden;
        padding: 27px 30px;
        border-radius: 22px;
        color: white;
        background: linear-gradient(120deg, #111c3b 0%, #1e40af 53%, #3b82f6 100%);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.18);
        margin-bottom: 23px;
    }

    .dashboard-hero::before {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        right: -45px;
        top: -135px;
        background: rgba(255,255,255,0.10);
    }

    .dashboard-hero::after {
        content: "";
        position: absolute;
        width: 170px;
        height: 170px;
        border-radius: 50%;
        right: 100px;
        bottom: -125px;
        background: rgba(147,197,253,0.15);
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .hero-kicker {
        display: inline-block;
        padding: 5px 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #dbeafe;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .hero-title {
        margin: 0;
        color: #ffffff;
        font-size: clamp(25px, 3vw, 35px);
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        margin-top: 9px;
        color: #dbeafe;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Section headings */
    .section-heading {
        color: #172033;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.35px;
        margin: 10px 0 3px 0;
    }

    .section-caption {
        color: #718096;
        font-size: 12px;
        margin-bottom: 14px;
    }

    /* Metric cards */
    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 118px;
        padding: 19px 21px;
        border-radius: 17px;
        border: 1px solid #e6ecf5;
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        box-shadow: 0 5px 15px rgba(15,23,42,0.035);
    }

    .metric-card::after {
        content: "";
        position: absolute;
        right: -19px;
        bottom: -37px;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: rgba(59,130,246,0.055);
    }

    .metric-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        margin-top: 11px;
        color: #172033;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.8px;
    }

    .metric-foot {
        margin-top: 5px;
        color: #94a3b8;
        font-size: 11px;
    }

    /* Filter labels */
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #334155;
        font-weight: 700;
        font-size: 12px;
    }

    /* Native expander */
    div[data-testid="stExpander"] {
        border: 1px solid #e3eaf4 !important;
        border-radius: 13px !important;
        background: #ffffff !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        padding: 10px 13px !important;
        color: #334155 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: #f5f8ff !important;
    }

    /* Streamlit buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.18s ease;
    }

    div.stButton > button:hover {
        border-color: #93b4ff;
        color: #1d4ed8;
    }

    /* Pagination */
    .page-indicator {
        text-align: center;
        padding: 8px;
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
    }

    /* Small status indicator */
    .result-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #eaf2ff;
        color: #1d4ed8;
        font-size: 11px;
        font-weight: 800;
    }

    .dashboard-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 11px;
        padding-top: 22px;
        padding-bottom: 8px;
    }

    /* Reduce excess vertical space */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.65rem;
    }

    @media (max-width: 768px) {
        .dashboard-hero {
            padding: 22px 20px;
            border-radius: 17px;
        }
    }
</style>
""")


# =========================================================
# HEADER
# =========================================================
st.html("""
<div class="dashboard-hero">
    <div class="hero-content">
        <div class="hero-kicker">⚡ SOLAR PROJECTS • CONTROL CENTER</div>
        <h1 class="hero-title">Intimation Dashboard</h1>
        <div class="hero-subtitle">
            Manage PSS information, review site details and track intimation records
            from one clean workspace.
        </div>
    </div>
</div>
""")


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_column(name):
    """Normalize column names for flexible matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns, candidates):
    """Find a column by trying preferred names first."""
    normalized = {
        normalize_column(col): col
        for col in columns
    }

    for candidate in candidates:
        key = normalize_column(candidate)
        if key in normalized:
            return normalized[key]

    # Fallback: allow a candidate to be contained in a column name.
    for candidate in candidates:
        key = normalize_column(candidate)
        if not key:
            continue

        for col in columns:
            if key in normalize_column(col):
                return col

    return None


def find_priority_column(columns):
    """Apply the requested priority-column preference."""
    preferred = [
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ]

    normalized = {
        normalize_column(col): col
        for col in columns
    }

    for name in preferred:
        key = normalize_column(name)
        if key in normalized:
            return normalized[key]

    for col in columns:
        if "priority" in str(col).lower():
            return col

    return None


def clean_text(value):
    if pd.isna(value):
        return "—"

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "nat"}:
        return "—"

    return text


def numeric_value(value):
    """Convert Excel numeric values safely."""
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    text = text.replace(",", "").replace("MW", "").strip()

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return 0.0

    try:
        return float(match.group())
    except (ValueError, TypeError):
        return 0.0


def display_capacity(value):
    if pd.isna(value):
        return "—"

    number = numeric_value(value)

    if number.is_integer():
        return f"{int(number):,} MW"

    return f"{number:,.2f} MW"


def format_number(number):
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def render_copy_button(pss_name, card_key):
    """
    Render a compact copy button.
    The iframe is used only for clipboard interaction, not card layout.
    """
    safe_value = json.dumps(str(pss_name))

    copy_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                padding: 2px 0;
                font-family: Arial, sans-serif;
            }}
            button {{
                width: 100%;
                height: 38px;
                border: 1px solid #dbeafe;
                border-radius: 11px;
                background: linear-gradient(135deg, #eff6ff, #dbeafe);
                color: #1d4ed8;
                font-size: 17px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.15s ease;
            }}
            button:hover {{
                background: #dbeafe;
                border-color: #93c5fd;
            }}
        </style>
    </head>
    <body>
        <button id="copyBtn" title="Copy PSS name">⚡</button>
        <script>
            const pssName = {safe_value};
            const btn = document.getElementById("copyBtn");

            btn.addEventListener("click", async function() {{
                try {{
                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                        await navigator.clipboard.writeText(pssName);
                    }} else {{
                        const area = document.createElement("textarea");
                        area.value = pssName;
                        area.style.position = "fixed";
                        area.style.opacity = "0";
                        document.body.appendChild(area);
                        area.focus();
                        area.select();
                        const copied = document.execCommand("copy");
                        area.remove();
                        if (!copied) throw new Error("Copy failed");
                    }}

                    btn.textContent = "✓";
                    btn.style.background = "#dcfce7";
                    btn.style.color = "#15803d";
                    btn.style.borderColor = "#86efac";

                    setTimeout(() => {{
                        btn.textContent = "⚡";
                        btn.style.background = "";
                        btn.style.color = "";
                        btn.style.borderColor = "";
                    }}, 1200);
                }} catch (err) {{
                    btn.textContent = "↗";
                    btn.title = "Clipboard unavailable in this browser";
                }}
            }});
        </script>
    </body>
    </html>
    """

    components.html(copy_html, height=43, scrolling=False)


def detail_item(label, value, icon="•"):
    safe_label = html.escape(str(label))
    safe_value = html.escape(clean_text(value))

    st.html(f"""
    <div style="
        padding: 11px 12px;
        border: 1px solid #e9eef6;
        border-radius: 11px;
        background: linear-gradient(145deg, #ffffff, #f8faff);
        min-height: 65px;
        overflow-wrap: anywhere;
        height: 100%;
    ">
        <div style="
            color: #94a3b8;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.65px;
            font-weight: 800;
            margin-bottom: 7px;
        ">{icon} {safe_label}</div>
        <div style="
            color: #26354d;
            font-size: 12px;
            line-height: 1.45;
            font-weight: 700;
        ">{safe_value}</div>
    </div>
    """)


def render_pss_card(row, index):
    pss_name = clean_text(row.get("PSS"))
    capacity = display_capacity(row.get("Capacity (MW)"))
    state = clean_text(row.get("State"))
    poc1 = clean_text(row.get("POC1"))

    # Main card wrapper uses native Streamlit layout.
    with st.container(border=True):
        # Decorative top strip
        st.html("""
        <div style="
            height: 4px;
            margin: -1px -1px 13px -1px;
            border-radius: 10px;
            background: linear-gradient(90deg, #2563eb, #60a5fa, #a5b4fc);
        "></div>
        """)

        # Title and lightning button
        title_col, copy_col = st.columns([5.3, 0.8], vertical_alignment="center")

        with title_col:
            st.html(f"""
            <div style="padding: 1px 0 7px 0;">
                <div style="
                    color: #94a3b8;
                    font-size: 9px;
                    letter-spacing: 1.1px;
                    font-weight: 800;
                    text-transform: uppercase;
                    margin-bottom: 5px;
                ">POWER SUB-STATION</div>
                <div style="
                    color: #172033;
                    font-size: 17px;
                    line-height: 1.35;
                    font-weight: 800;
                    letter-spacing: -0.35px;
                    overflow-wrap: anywhere;
                ">{html.escape(pss_name)}</div>
            </div>
            """)

        with copy_col:
            render_copy_button(pss_name, index)

        # Capacity badge and card metadata
        st.html(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            flex-wrap: wrap;
            margin: 2px 0 12px 0;
        ">
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 7px 10px;
                background: #ecfdf5;
                border: 1px solid #d1fae5;
                border-radius: 9px;
                color: #047857;
                font-size: 12px;
                font-weight: 800;
            ">⚡ {html.escape(capacity)}</span>

            <span style="
                display: inline-block;
                padding: 6px 9px;
                border-radius: 9px;
                background: #f1f5f9;
                color: #475569;
                font-size: 10px;
                font-weight: 700;
                max-width: 100%;
                overflow-wrap: anywhere;
            ">📍 {html.escape(state)}</span>
        </div>
        """)

        # Contact detail
        st.html(f"""
        <div style="
            border: 1px solid #e8eef7;
            border-radius: 11px;
            padding: 10px 12px;
            background: #fbfdff;
            margin-bottom: 11px;
        ">
            <div style="
                color: #94a3b8;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 0.7px;
                margin-bottom: 5px;
            ">PRIMARY CONTACT · POC 1</div>
            <div style="
                color: #334155;
                font-size: 12px;
                font-weight: 700;
                overflow-wrap: anywhere;
            ">{html.escape(poc1)}</div>
        </div>
        """)

        # Actual Streamlit expander: details participate in page layout.
        with st.expander("⌄  View complete site details", expanded=False):
            st.html("""
            <div style="
                color: #1e40af;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0.8px;
                margin: 2px 0 11px 0;
            ">SITE INFORMATION</div>
            """)

            priority = clean_text(row.get("Priority"))
            poc2 = clean_text(row.get("POC 2"))
            poc3 = clean_text(row.get("POC 3"))
            latitude = clean_text(row.get("Lat"))
            longitude = clean_text(row.get("Long"))
            check_status = clean_text(row.get("Check Status"))

            detail_columns = st.columns(2, gap="small")

            details = [
                ("Priority", priority, "✦"),
                ("POC 2", poc2, "☎"),
                ("POC 3", poc3, "☎"),
                ("Latitude", latitude, "◎"),
                ("Longitude", longitude, "◎"),
                ("Check Status", check_status, "✓"),
            ]

            for i, (label, value, icon) in enumerate(details):
                with detail_columns[i % 2]:
                    detail_item(label, value, icon)

            st.html("""
            <div style="
                margin-top: 12px;
                padding-top: 10px;
                border-top: 1px solid #e9eef6;
                color: #94a3b8;
                font-size: 10px;
            ">End of site details</div>
            """)


# =========================================================
# LOAD EXCEL
# =========================================================
@st.cache_data(show_spinner="Loading intimation records...")
def load_data(file_path, modified_time):
    # modified_time is included so an updated Excel file refreshes the cache.
    del modified_time

    return pd.read_excel(file_path)


file_path = Path(__file__).resolve().parent.parent / "intimation.xlsx"

if not file_path.exists():
    st.error(
        "Excel file 'intimation.xlsx' nahi mili. "
        "Please check karein ki file project root folder mein hai."
    )
    st.stop()

try:
    raw_df = load_data(str(file_path), file_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Excel file load nahi ho paayi: {exc}")
    st.stop()

if raw_df.empty:
    st.warning("Excel file mein abhi koi record available nahi hai.")
    st.stop()


# =========================================================
# FLEXIBLE COLUMN MAPPING
# =========================================================
columns = list(raw_df.columns)

column_map = {
    "PSS": find_column(
        columns,
        ["PSS", "PSS Name", "Substation", "Sub Station", "Name of PSS"],
    ),
    "State": find_column(
        columns,
        ["State", "State Name"],
    ),
    "Capacity (MW)": find_column(
        columns,
        ["Capacity (MW)", "Capacity MW", "Capacity", "MW"],
    ),
    "POC1": find_column(
        columns,
        ["POC1", "POC 1", "POC-1", "POC 1 Name", "Primary Contact"],
    ),
    "POC 2": find_column(
        columns,
        ["POC 2", "POC2", "POC-2", "POC 2 Name"],
    ),
    "POC 3": find_column(
        columns,
        ["POC 3", "POC3", "POC-3", "POC 3 Name"],
    ),
    "Lat": find_column(
        columns,
        ["Lat", "Latitude"],
    ),
    "Long": find_column(
        columns,
        ["Long", "Longitude", "Lon", "Lng"],
    ),
    "Check Status": find_column(
        columns,
        ["Check Status", "Status", "Intimation Status"],
    ),
}

column_map["Priority"] = find_priority_column(columns)

if column_map["PSS"] is None:
    st.error(
        "PSS column identify nahi ho paaya. "
        "Excel mein PSS name wala column check karein."
    )
    st.caption("Available columns: " + ", ".join(map(str, columns)))
    st.stop()


# Create a clean display dataframe while preserving all source rows.
df = pd.DataFrame(index=raw_df.index)

for display_name, source_name in column_map.items():
    if source_name is None:
        df[display_name] = "—"
    else:
        df[display_name] = raw_df[source_name]

df = df.reset_index(drop=True)
df["_row_id"] = df.index

# Normalize values used for searching and display.
for col in ["PSS", "State", "POC1", "POC 2", "POC 3", "Priority", "Check Status"]:
    df[col] = df[col].map(clean_text)

df["Capacity (MW)"] = df["Capacity (MW)"].map(numeric_value)


# =========================================================
# FILTERS
# =========================================================
st.html("""
<div class="section-heading">Find your PSS</div>
<div class="section-caption">
Search and narrow down records using connected filters.
</div>
""")

filter1, filter2, filter3 = st.columns([1.6, 1, 1], gap="medium")

with filter1:
    pss_search = st.text_input(
        "Search PSS",
        placeholder="Type PSS name...",
        key="intimation_pss_search",
    )

# PSS search filters the available POC1 choices.
if pss_search.strip():
    pss_mask = df["PSS"].str.contains(
        pss_search.strip(),
        case=False,
        na=False,
        regex=False,
    )
    search_df = df[pss_mask].copy()
else:
    search_df = df.copy()

poc1_options = sorted(
    value for value in search_df["POC1"].unique()
    if value != "—"
)

with filter2:
    selected_poc1 = st.selectbox(
        "POC 1",
        options=["All POC 1"] + poc1_options,
        key="intimation_poc1_filter",
    )

# State choices depend on PSS search and the selected POC1.
if selected_poc1 != "All POC 1":
    state_source_df = search_df[
        search_df["POC1"] == selected_poc1
    ].copy()
else:
    state_source_df = search_df.copy()

state_options = sorted(
    value for value in state_source_df["State"].unique()
    if value != "—"
)

with filter3:
    selected_state = st.selectbox(
        "State",
        options=["All States"] + state_options,
        key="intimation_state_filter",
    )


# =========================================================
# APPLY FILTERS
# =========================================================
filtered_df = search_df.copy()

if selected_poc1 != "All POC 1":
    filtered_df = filtered_df[
        filtered_df["POC1"] == selected_poc1
    ]

if selected_state != "All States":
    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]

filtered_df = filtered_df.reset_index(drop=True)


# =========================================================
# OVERVIEW METRICS
# =========================================================
total_records = len(filtered_df)
total_capacity = filtered_df["Capacity (MW)"].sum()

metric_col1, metric_col2 = st.columns(2, gap="medium")

with metric_col1:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">▦ &nbsp; Matching Records</div>
        <div class="metric-value">{total_records:,}</div>
        <div class="metric-foot">PSS records matching your filters</div>
    </div>
    """)

with metric_col2:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">⚡ &nbsp; Total Capacity</div>
        <div class="metric-value">{format_number(total_capacity)} <span style="font-size:15px;color:#64748b;">MW</span></div>
        <div class="metric-foot">Combined capacity of matching records</div>
    </div>
    """)


# =========================================================
# RESULTS HEADING
# =========================================================
st.html("""
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:8px;
    margin-top:25px;
    margin-bottom:3px;
">
    <div class="section-heading" style="margin:0;">PSS Directory</div>
</div>
<div class="section-caption">
Open any card to view full site information. Use ⚡ to copy a PSS name.
</div>
""")


# =========================================================
# PAGINATION
# =========================================================
PAGE_SIZE = 20
total_pages = max(1, (total_records + PAGE_SIZE - 1) // PAGE_SIZE)

if "intimation_page" not in st.session_state:
    st.session_state["intimation_page"] = 1

# Keep the current page within valid bounds after filtering.
if st.session_state["intimation_page"] > total_pages:
    st.session_state["intimation_page"] = total_pages

if st.session_state["intimation_page"] < 1:
    st.session_state["intimation_page"] = 1

if total_records > 0:
    page_start = (st.session_state["intimation_page"] - 1) * PAGE_SIZE
    page_df = filtered_df.iloc[page_start:page_start + PAGE_SIZE]

    page_number_col, page_info_col = st.columns([1, 2], vertical_alignment="center")

    with page_number_col:
        st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            key="intimation_page_selector",
            index=st.session_state["intimation_page"] - 1,
            on_change=lambda: st.session_state.update(
                intimation_page=st.session_state["intimation_page_selector"]
            ),
        )

    with page_info_col:
        first_record = page_start + 1
        last_record = min(page_start + PAGE_SIZE, total_records)

        st.html(f"""
        <div style="
            text-align:right;
            color:#64748b;
            font-size:12px;
            padding-top:20px;
        ">
            Showing <b style="color:#1e293b;">{first_record}–{last_record}</b>
            of <b style="color:#1e293b;">{total_records}</b> records
        </div>
        """)

    # =====================================================
    # CARD GRID: TWO COLUMNS
    # =====================================================
    card_columns = st.columns(2, gap="medium")

    for position, (_, row) in enumerate(page_df.iterrows()):
        with card_columns[position % 2]:
            render_pss_card(
                row,
                index=f"{int(row['_row_id'])}",
            )

    # =====================================================
    # BOTTOM PAGINATION
    # =====================================================
    st.divider()

    previous_col, page_label_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        st.button(
            "← Previous",
            key="intimation_previous",
            disabled=st.session_state["intimation_page"] <= 1,
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                intimation_page=max(
                    1,
                    st.session_state["intimation_page"] - 1,
                )
            ),
        )

    with page_label_col:
        st.markdown(
            f'<div class="page-indicator">Page '
            f'{st.session_state["intimation_page"]} of {total_pages}</div>',
            unsafe_allow_html=True,
        )

    with next_col:
        st.button(
            "Next →",
            key="intimation_next",
            disabled=st.session_state["intimation_page"] >= total_pages,
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                intimation_page=min(
                    total_pages,
                    st.session_state["intimation_page"] + 1,
                )
            ),
        )

else:
    st.html("""
    <div style="
        margin-top:10px;
        padding:35px 20px;
        border:1px dashed #cbd5e1;
        border-radius:17px;
        text-align:center;
        background:#ffffff;
    ">
        <div style="font-size:28px;margin-bottom:9px;">🔎</div>
        <div style="color:#1e293b;font-weight:800;font-size:16px;">
            No matching PSS records
        </div>
        <div style="color:#64748b;font-size:12px;margin-top:6px;">
            Try changing your search or selecting different filters.
        </div>
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================
st.html("""
<div class="dashboard-footer">
    INTIMATION DASHBOARD &nbsp; • &nbsp; PSS MANAGEMENT WORKSPACE
</div>
""")
