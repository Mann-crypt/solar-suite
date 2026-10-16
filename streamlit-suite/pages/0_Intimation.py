
from pathlib import Path
import html
import json
import re

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
# CLEAN, SPACIOUS UI
# =========================================================

st.html("""
<style>
    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.15rem;
        padding-bottom: 1.5rem;
    }

    /* Header */
    .dashboard-header {
        padding: 19px 22px;
        margin-bottom: 18px;
        border-radius: 15px;
        background: linear-gradient(115deg, #172554, #1d4ed8 70%, #3b82f6);
        color: white;
        box-shadow: 0 5px 16px rgba(37, 99, 235, .12);
    }

    .header-label {
        display: inline-block;
        padding: 4px 9px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 20px;
        background: rgba(255,255,255,.10);
        color: #dbeafe;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .dashboard-header h1 {
        margin: 0;
        color: white;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -.5px;
    }

    .dashboard-header p {
        margin: 5px 0 0;
        color: #dbeafe;
        font-size: 12px;
    }

    /* Section heading */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 18px 0 10px;
        color: #1e293b;
        font-size: 15px;
        font-weight: 800;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: #e8efff;
        color: #1d4ed8;
        font-size: 14px;
    }

    .filter-caption {
        margin: 0 0 8px;
        color: #64748b;
        font-size: 11px;
    }

    /* Overview metrics */
    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 91px;
        padding: 14px 17px;
        border-radius: 13px;
        color: white;
        box-shadow: 0 4px 12px rgba(15,23,42,.07);
    }

    .metric-records {
        background: linear-gradient(120deg, #1d4ed8, #3b82f6);
    }

    .metric-capacity {
        background: linear-gradient(120deg, #047857, #10b981);
    }

    .metric-label {
        color: rgba(255,255,255,.88);
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 6px;
        color: white;
        font-size: 27px;
        line-height: 1.2;
        font-weight: 800;
    }

    .metric-icon {
        position: absolute;
        top: 13px;
        right: 15px;
        font-size: 23px;
        opacity: .65;
    }

    /* Card wrapper */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e1e8f2 !important;
        border-radius: 13px !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(15,23,42,.035);
        transition: border-color .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #b9cef8 !important;
        box-shadow: 0 5px 13px rgba(37,99,235,.07);
    }

    /* Card title */
    .pss-title-row {
        display: flex;
        align-items: center;
        gap: 9px;
        min-width: 0;
        margin-bottom: 12px;
    }

    .pss-icon {
        display: flex;
        flex: 0 0 34px;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: #eff6ff;
        border: 1px solid #dbeafe;
        font-size: 16px;
    }

    .pss-heading {
        color: #172554;
        font-size: 13px;
        font-weight: 800;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .pss-subheading {
        margin-top: 2px;
        color: #94a3b8;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .65px;
        text-transform: uppercase;
    }

    /* Compact information */
    .pss-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 8px;
    }

    .pss-info {
        min-width: 0;
        padding: 9px 10px;
        border: 1px solid #edf1f7;
        border-radius: 9px;
        background: #f8fafc;
    }

    .pss-info-label {
        margin-bottom: 4px;
        color: #94a3b8;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .65px;
        text-transform: uppercase;
    }

    .pss-info-value {
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .capacity-tile {
        background: #eff6ff;
        border-color: #dbeafe;
    }

    .capacity-tile .pss-info-value {
        color: #1d4ed8;
    }

    .state-tile {
        background: #f0fdf4;
        border-color: #dcfce7;
    }

    .state-tile .pss-info-value {
        color: #047857;
    }

    /* Expandable details */
    div[data-testid="stExpander"] {
        margin-top: 8px;
        border: 1px solid #e7edf6 !important;
        border-radius: 9px !important;
        background: #fbfdff !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        padding-top: 7px !important;
        padding-bottom: 7px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }

    .detail-tile {
        min-height: 48px;
        margin-bottom: 7px;
        padding: 9px;
        border: 1px solid #e7edf5;
        border-radius: 8px;
        background: #f8fafc;
    }

    .detail-label {
        margin-bottom: 4px;
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .detail-value {
        color: #1e293b;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        max-width: 100%;
        padding: 4px 8px;
        border-radius: 6px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 10px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    /* Empty state */
    .empty-state {
        padding: 24px 15px;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        background: white;
        color: #64748b;
        text-align: center;
    }

    .empty-state strong {
        display: block;
        margin-bottom: 5px;
        color: #334155;
        font-size: 15px;
    }

    .results-caption {
        margin: 2px 0 11px;
        color: #64748b;
        font-size: 11px;
    }

    .page-label {
        padding-top: 7px;
        color: #64748b;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
    }

    /* Moderate spacing, not overly compact */
    @media (max-width: 700px) {
        .block-container {
            padding-left: .7rem;
            padding-right: .7rem;
        }

        .dashboard-header {
            padding: 16px;
        }

        .dashboard-header h1 {
            font-size: 23px;
        }

        .metric-value {
            font-size: 23px;
        }
    }
</style>
""")

<style>
    /* Theme-aware colors from Streamlit */
    :root {
        color-scheme: light dark;
    }

    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.15rem;
        padding-bottom: 1.5rem;
    }

    /* Header keeps its gradient in both themes */
    .dashboard-header {
        padding: 19px 22px;
        margin-bottom: 18px;
        border-radius: 15px;
        background: linear-gradient(115deg, #172554, #1d4ed8 70%, #3b82f6);
        color: #ffffff;
        box-shadow: 0 5px 16px rgba(37, 99, 235, .15);
    }

    .header-label {
        display: inline-block;
        padding: 4px 9px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 20px;
        background: rgba(255,255,255,.10);
        color: #dbeafe;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .dashboard-header h1 {
        margin: 0;
        color: #ffffff !important;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -.5px;
    }

    .dashboard-header p {
        margin: 5px 0 0;
        color: #dbeafe !important;
        font-size: 12px;
    }

    /* Section headings */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 18px 0 10px;
        color: var(--text-color);
        font-size: 15px;
        font-weight: 800;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: var(--secondary-background-color);
        color: #60a5fa;
        font-size: 14px;
    }

    .filter-caption,
    .results-caption {
        color: var(--text-color);
        opacity: .72;
        font-size: 11px;
    }

    .filter-caption {
        margin: 0 0 8px;
    }

    .results-caption {
        margin: 2px 0 11px;
    }

    /* Overview cards */
    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 91px;
        padding: 14px 17px;
        border-radius: 13px;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(15,23,42,.12);
    }

    .metric-records {
        background: linear-gradient(120deg, #1d4ed8, #3b82f6);
    }

    .metric-capacity {
        background: linear-gradient(120deg, #047857, #10b981);
    }

    .metric-label {
        color: #eaf2ff !important;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 6px;
        color: #ffffff !important;
        font-size: 27px;
        line-height: 1.2;
        font-weight: 800;
    }

    .metric-icon {
        position: absolute;
        top: 13px;
        right: 15px;
        color: #ffffff;
        font-size: 23px;
        opacity: .65;
    }

    /* Cards adapt to active Streamlit theme */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border-color) !important;
        border-radius: 13px !important;
        background: var(--secondary-background-color) !important;
        box-shadow: 0 2px 8px rgba(15,23,42,.06);
        transition: border-color .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #6095ef !important;
        box-shadow: 0 5px 13px rgba(37,99,235,.12);
    }

    /* PSS title */
    .pss-title-row {
        display: flex;
        align-items: center;
        gap: 9px;
        min-width: 0;
        margin-bottom: 12px;
    }

    .pss-icon {
        display: flex;
        flex: 0 0 34px;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        font-size: 16px;
    }

    .pss-heading {
        color: var(--text-color) !important;
        font-size: 13px;
        font-weight: 800;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .pss-subheading {
        margin-top: 2px;
        color: var(--text-color);
        opacity: .6;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .65px;
        text-transform: uppercase;
    }

    /* Information tiles */
    .pss-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 8px;
    }

    .pss-info {
        min-width: 0;
        padding: 9px 10px;
        border: 1px solid var(--border-color);
        border-radius: 9px;
        background: var(--background-color);
    }

    .pss-info-label {
        margin-bottom: 4px;
        color: var(--text-color);
        opacity: .65;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .65px;
        text-transform: uppercase;
    }

    .pss-info-value {
        color: var(--text-color) !important;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    /* Keep these accent tiles readable in either theme */
    .capacity-tile {
        background: color-mix(in srgb, #3b82f6 11%, var(--secondary-background-color));
        border-color: color-mix(in srgb, #3b82f6 30%, var(--border-color));
    }

    .capacity-tile .pss-info-value {
        color: #60a5fa !important;
    }

    .state-tile {
        background: color-mix(in srgb, #10b981 10%, var(--secondary-background-color));
        border-color: color-mix(in srgb, #10b981 27%, var(--border-color));
    }

    .state-tile .pss-info-value {
        color: #34d399 !important;
    }

    /* Expanded details */
    div[data-testid="stExpander"] {
        margin-top: 8px;
        border: 1px solid var(--border-color) !important;
        border-radius: 9px !important;
        background: var(--secondary-background-color) !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        color: var(--text-color) !important;
        padding-top: 7px !important;
        padding-bottom: 7px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }

    .detail-tile {
        min-height: 48px;
        margin-bottom: 7px;
        padding: 9px;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        background: var(--background-color);
    }

    .detail-label {
        margin-bottom: 4px;
        color: var(--text-color);
        opacity: .7;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .detail-value {
        color: var(--text-color) !important;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        max-width: 100%;
        padding: 4px 8px;
        border-radius: 6px;
        background: color-mix(in srgb, #3b82f6 18%, var(--secondary-background-color));
        color: #60a5fa;
        font-size: 10px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    /* Copy button */
    div[data-testid="stVerticalBlock"] iframe {
        border: 0;
    }

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 9px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    /* Empty state */
    .empty-state {
        padding: 24px 15px;
        border: 1px dashed var(--border-color);
        border-radius: 12px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        text-align: center;
    }

    .empty-state strong {
        display: block;
        margin-bottom: 5px;
        color: var(--text-color);
        font-size: 15px;
    }

    .page-label {
        padding-top: 7px;
        color: var(--text-color);
        opacity: .75;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: .7rem;
            padding-right: .7rem;
        }

        .dashboard-header {
            padding: 16px;
        }

        .dashboard-header h1 {
            font-size: 23px;
        }

        .metric-value {
            font-size: 23px;
        }
    }
</style>
""")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_column(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(columns, candidates=None, startswith=None):
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    if candidates:
        for candidate in candidates:
            key = normalize_column(candidate)
            if key in normalized:
                return normalized[key]

    if startswith:
        prefix = normalize_column(startswith)

        for column in columns:
            if normalize_column(column).startswith(prefix):
                return column

    return None


def clean_text(value, default="—"):
    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "nat"}:
        return default

    return value


def safe_html(value):
    return html.escape(clean_text(value))


def numeric_value(value):
    if pd.isna(value):
        return 0.0

    try:
        value = str(value).strip().replace(",", "")
        value = re.sub(r"(?i)\s*mw\s*$", "", value)

        match = re.search(r"-?\d+(?:\.\d+)?", value)

        return float(match.group()) if match else 0.0

    except (ValueError, TypeError):
        return 0.0


def display_capacity(value):
    number = numeric_value(value)

    if number.is_integer():
        return f"{int(number):,} MW"

    return f"{number:,.2f}".rstrip("0").rstrip(".") + " MW"


def unique_values(series):
    values = {
        clean_text(value)
        for value in series.tolist()
        if clean_text(value) != "—"
    }

    return sorted(values, key=str.casefold)


def detail_item(label, value, status=False):
    value_html = safe_html(value)

    if status:
        value_html = (
            f'<span class="status-pill">{value_html}</span>'
        )

    st.markdown(
        f"""
        <div class="detail-tile">
            <div class="detail-label">{safe_html(label)}</div>
            <div class="detail-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(pss_name, card_key):
    # Separate iframe keeps copy interaction isolated from the expander.
    js_text = json.dumps(str(pss_name))

    components.html(
        f"""
        <div style="
            display:flex;
            justify-content:flex-end;
            padding:0;
            margin:0;
            font-family:Arial,sans-serif;
        ">
            <button id="copy_{card_key}" title="Copy PSS name"
                style="
                    height:32px;
                    min-width:38px;
                    cursor:pointer;
                    border:1px solid #dbeafe;
                    border-radius:9px;
                    background:#eff6ff;
                    color:#1d4ed8;
                    font-size:16px;
                ">⚡</button>
            <span id="msg_{card_key}" style="
                display:none;
                margin:8px 0 0 5px;
                color:#047857;
                font-size:10px;
                font-weight:700;
            ">Copied</span>
        </div>

        <script>
        const btn = document.getElementById("copy_{card_key}");
        const msg = document.getElementById("msg_{card_key}");
        const pssName = {js_text};

        btn.addEventListener("click", async () => {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(pssName);
                }} else {{
                    const el = document.createElement("textarea");
                    el.value = pssName;
                    el.style.position = "fixed";
                    el.style.left = "-9999px";
                    document.body.appendChild(el);
                    el.select();
                    const ok = document.execCommand("copy");
                    el.remove();

                    if (!ok) throw new Error("Copy failed");
                }}

                btn.textContent = "✓";
                msg.style.display = "inline";

                setTimeout(() => {{
                    btn.textContent = "⚡";
                    msg.style.display = "none";
                }}, 1200);
            }} catch (e) {{
                btn.textContent = "!";
                setTimeout(() => btn.textContent = "⚡", 1200);
            }}
        }});
        </script>
        """,
        height=38,
        scrolling=False,
    )


def render_pss_card(row, columns, card_key):
    pss_name = clean_text(row[columns["pss"]])

    state = (
        clean_text(row[columns["state"]])
        if columns["state"] else "—"
    )

    capacity = (
        display_capacity(row[columns["capacity"]])
        if columns["capacity"] else "—"
    )

    poc1 = (
        clean_text(row[columns["poc1"]])
        if columns["poc1"] else "—"
    )

    with st.container(border=True):
        title_col, copy_col = st.columns(
            [5.5, 0.8],
            vertical_alignment="center",
            gap="small",
        )

        with title_col:
            st.markdown(
                f"""
                <div class="pss-title-row">
                    <div class="pss-icon">⚡</div>
                    <div>
                        <div class="pss-heading">{safe_html(pss_name)}</div>
                        <div class="pss-subheading">Power Substation</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            render_copy_button(pss_name, card_key)

        st.markdown(
            f"""
            <div class="pss-info-grid">
                <div class="pss-info capacity-tile">
                    <div class="pss-info-label">Capacity</div>
                    <div class="pss-info-value">{safe_html(capacity)}</div>
                </div>
                <div class="pss-info state-tile">
                    <div class="pss-info-label">State</div>
                    <div class="pss-info-value">{safe_html(state)}</div>
                </div>
            </div>

            <div class="pss-info">
                <div class="pss-info-label">POC 1</div>
                <div class="pss-info-value">{safe_html(poc1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View site details", expanded=False):
            left, right = st.columns(2, gap="small")

            with left:
                detail_item(
                    "Priority",
                    row[columns["priority"]]
                    if columns["priority"] else "—",
                )
                detail_item(
                    "POC 2",
                    row[columns["poc2"]]
                    if columns["poc2"] else "—",
                )
                detail_item(
                    "Latitude",
                    row[columns["lat"]]
                    if columns["lat"] else "—",
                )

            with right:
                detail_item(
                    "POC 3",
                    row[columns["poc3"]]
                    if columns["poc3"] else "—",
                )
                detail_item(
                    "Longitude",
                    row[columns["long"]]
                    if columns["long"] else "—",
                )
                detail_item(
                    "Check Status",
                    row[columns["status"]]
                    if columns["status"] else "—",
                    status=True,
                )


# =========================================================
# LOAD EXCEL
# =========================================================

@st.cache_data(show_spinner="Loading intimation data...")
def load_data(file_path, modified_time):
    return pd.read_excel(file_path)


file_path = Path(__file__).resolve().parent.parent / "intimation.xlsx"

if not file_path.exists():
    st.error(
        f"Excel file not found: {file_path}. "
        "Ensure intimation.xlsx is present in the project root."
    )
    st.stop()

try:
    df = load_data(str(file_path), file_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to read intimation.xlsx: {exc}")
    st.stop()

if df.empty:
    st.warning("The Excel file contains no records.")
    st.stop()


# =========================================================
# COLUMN MAPPING
# =========================================================

all_columns = list(df.columns)

columns = {
    "pss": find_column(
        all_columns,
        [
            "PSS",
            "PSS Name",
            "Name of PSS",
            "Substation",
            "Substation Name",
        ],
    ),
    "state": find_column(
        all_columns,
        ["State", "PSS State", "State Name"],
    ),
    "capacity": find_column(
        all_columns,
        [
            "Capacity (MW)",
            "Capacity MW",
            "Capacity",
            "PSS Capacity (MW)",
        ],
    ),
    "poc1": find_column(
        all_columns,
        ["POC1", "POC 1", "POC-1"],
    ),
    "poc2": find_column(
        all_columns,
        ["POC 2", "POC2", "POC-2"],
    ),
    "poc3": find_column(
        all_columns,
        ["POC 3", "POC3", "POC-3"],
    ),
    "lat": find_column(
        all_columns,
        ["Lat", "Latitude"],
    ),
    "long": find_column(
        all_columns,
        ["Long", "Longitude", "Lon"],
    ),
    "status": find_column(
        all_columns,
        ["Check Status", "Status", "Intimation Status"],
    ),
}

columns["priority"] = find_column(
    all_columns,
    [
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ],
)

if columns["priority"] is None:
    columns["priority"] = find_column(
        all_columns,
        startswith="Priority",
    )

if columns["pss"] is None:
    st.error("PSS column could not be identified in the Excel file.")
    st.write("Available columns:", all_columns)
    st.stop()


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-label">SOLAR SUITE · OPERATIONS</div>
        <h1>⚡ Intimation Dashboard</h1>
        <p>Search PSS sites, refine filters and view site details.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FILTERS
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">⌕</span>
        Search & Filters
    </div>
    <div class="filter-caption">
        Start typing a PSS name to see suggestions. POC 1 and State
        options update automatically based on your selections.
    </div>
    """,
    unsafe_allow_html=True,
)

# PSS suggestions are searchable directly inside the selectbox.
# A blank selection means all PSS sites.
pss_values = unique_values(df[columns["pss"]])

f1, f2, f3 = st.columns([2.0, 1.25, 1.25], gap="medium")

with f1:
    selected_pss = st.selectbox(
        "Search PSS",
        options=pss_values,
        index=None,
        placeholder="Type or select a PSS name...",
        key="intimation_pss_autocomplete",
    )

# Filter by the selected PSS first.
search_df = df.copy()

if selected_pss:
    search_df = search_df[
        search_df[columns["pss"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_pss
    ]

# POC1 suggestions depend on the selected PSS.
with f2:
    if columns["poc1"]:
        poc1_values = unique_values(search_df[columns["poc1"]])

        selected_poc1 = st.selectbox(
            "POC 1",
            options=poc1_values,
            index=None,
            placeholder="All POC 1",
            key="intimation_poc1_autocomplete",
        )
    else:
        selected_poc1 = None
        st.selectbox(
            "POC 1",
            options=[],
            placeholder="Column unavailable",
            disabled=True,
            key="intimation_poc1_unavailable",
        )

# State suggestions depend on both PSS and POC1.
state_source = search_df.copy()

if selected_poc1 and columns["poc1"]:
    state_source = state_source[
        state_source[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_poc1
    ]

with f3:
    if columns["state"]:
        state_values = unique_values(state_source[columns["state"]])

        selected_state = st.selectbox(
            "State",
            options=state_values,
            index=None,
            placeholder="All States",
            key="intimation_state_autocomplete",
        )
    else:
        selected_state = None
        st.selectbox(
            "State",
            options=[],
            placeholder="Column unavailable",
            disabled=True,
            key="intimation_state_unavailable",
        )

# Apply selections to the final result.
filtered_df = search_df.copy()

if selected_poc1 and columns["poc1"]:
    filtered_df = filtered_df[
        filtered_df[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_poc1
    ]

if selected_state and columns["state"]:
    filtered_df = filtered_df[
        filtered_df[columns["state"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == selected_state
    ]

filtered_df = filtered_df.reset_index(drop=True)


# =========================================================
# OVERVIEW
# =========================================================

record_count = len(filtered_df)

if columns["capacity"]:
    total_capacity = filtered_df[columns["capacity"]].apply(
        numeric_value
    ).sum()
    capacity_text = display_capacity(total_capacity)
else:
    capacity_text = "—"

metric1, metric2 = st.columns(2, gap="medium")

with metric1:
    st.markdown(
        f"""
        <div class="metric-card metric-records">
            <div class="metric-icon">▦</div>
            <div class="metric-label">Matching Records</div>
            <div class="metric-value">{record_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f"""
        <div class="metric-card metric-capacity">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Total Capacity</div>
            <div class="metric-value">{safe_html(capacity_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGINATION
# =========================================================

PAGE_SIZE = 20

filter_signature = (
    selected_pss or "",
    selected_poc1 or "",
    selected_state or "",
)

if st.session_state.get("intimation_filter_signature") != filter_signature:
    st.session_state["intimation_filter_signature"] = filter_signature
    st.session_state["intimation_page"] = 1

total_pages = max(1, (record_count + PAGE_SIZE - 1) // PAGE_SIZE)

current_page = st.session_state.get("intimation_page", 1)
current_page = min(max(current_page, 1), total_pages)

st.session_state["intimation_page"] = current_page

start_index = (current_page - 1) * PAGE_SIZE
end_index = min(start_index + PAGE_SIZE, record_count)

page_df = filtered_df.iloc[start_index:end_index]


# =========================================================
# PSS CARDS
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">▤</span>
        PSS Sites
    </div>
    """,
    unsafe_allow_html=True,
)

if record_count > 0:
    st.markdown(
        f"""
        <div class="results-caption">
            Showing <b>{start_index + 1}–{end_index}</b>
            of <b>{record_count:,}</b> sites
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = list(page_df.iterrows())

    for row_start in range(0, len(records), 2):
        card_columns = st.columns(2, gap="medium")

        for offset in range(2):
            position = row_start + offset

            if position >= len(records):
                continue

            original_index, row = records[position]

            with card_columns[offset]:
                render_pss_card(
                    row=row,
                    columns=columns,
                    card_key=f"site_{current_page}_{original_index}",
                )

        st.markdown(
            '<div style="height:5px"></div>',
            unsafe_allow_html=True,
        )

    if total_pages > 1:
        st.markdown("---")

        prev_col, page_col, next_col = st.columns(
            [1, 2, 1],
            vertical_alignment="center",
        )

        with prev_col:
            if st.button(
                "← Previous",
                use_container_width=True,
                disabled=current_page == 1,
                key="intimation_previous",
            ):
                st.session_state["intimation_page"] = current_page - 1
                st.rerun()

        with page_col:
            st.markdown(
                f'<div class="page-label">Page {current_page} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

        with next_col:
            if st.button(
                "Next →",
                use_container_width=True,
                disabled=current_page == total_pages,
                key="intimation_next",
            ):
                st.session_state["intimation_page"] = current_page + 1
                st.rerun()

else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>🔎 No matching PSS sites</strong>
            Try another PSS name or change the POC 1 or State selection.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        margin-top:22px;
        padding-top:12px;
        border-top:1px solid #e2e8f0;
        color:#94a3b8;
        text-align:center;
        font-size:10px;
    ">
        SOLAR SUITE · INTIMATION DASHBOARD
    </div>
    """,
    unsafe_allow_html=True,
)
