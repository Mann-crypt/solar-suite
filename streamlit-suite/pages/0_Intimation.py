
from pathlib import Path
from html import escape
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

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"
PAGE_SIZE = 20
ALL_USERS = "All Users"


# =========================================================
# HELPERS
# =========================================================
def clean(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def safe_html(value):
    return escape(clean(value), quote=True)


def normalize_header(value):
    value = str(value).strip().lower()
    value = re.sub(r"[\n\r\t]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def find_column(columns, candidates=(), starts_with=()):
    normalized = {col: normalize_header(col) for col in columns}

    for candidate in candidates:
        target = normalize_header(candidate)
        for col, value in normalized.items():
            if value == target:
                return col

    for prefix in starts_with:
        target = normalize_header(prefix)
        for col, value in normalized.items():
            if value.startswith(target):
                return col

    return None


def unique_sorted(series):
    values = [clean(v) for v in series.tolist()]
    return sorted({v for v in values if v}, key=str.casefold)


def format_capacity(value):
    try:
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return clean(value) or "0"


def reset_page():
    st.session_state["intimation_page"] = 0


def status_style(status):
    text = clean(status).lower()

    if any(word in text for word in (
        "complete", "done", "checked", "yes", "clear"
    )):
        return "status-good"

    if any(word in text for word in (
        "pending", "not", "hold", "no", "issue"
    )):
        return "status-pending"

    return "status-neutral"


BOLT_SVG = """
<svg viewBox="0 0 32 32" aria-hidden="true">
  <defs>
    <linearGradient id="boltFill" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fff5b8"/>
      <stop offset="48%" stop-color="#ffd044"/>
      <stop offset="100%" stop-color="#ff8a22"/>
    </linearGradient>
  </defs>
  <path d="M17.9 2.7 7 17.1h7l-1.1 12.2L25 13.7h-7.3l.2-11Z"
        fill="url(#boltFill)"
        stroke="#8278ff"
        stroke-width="1.1"
        stroke-linejoin="round"/>
</svg>
"""


# =========================================================
# LOAD EXCEL
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(file_path, modified_time):
    # File modification time invalidates the cache after Excel updates.
    del modified_time

    raw = pd.read_excel(file_path, engine="openpyxl")
    raw.columns = [str(c).strip() for c in raw.columns]
    columns = list(raw.columns)

    mapping = {
        "PSS": find_column(
            columns,
            ["PSS", "PSS Name", "Substation", "Power Substation"],
        ),
        "State": find_column(columns, ["State", "States"]),
        "Capacity (MW)": find_column(
            columns,
            ["Capacity (MW)", "Capacity MW", "Capacity", "MW"],
        ),
        "POC1": find_column(
            columns,
            ["POC1", "POC 1", "POC-1", "Point of Contact 1"],
        ),
        "POC 2": find_column(
            columns,
            ["POC 2", "POC2", "POC-2", "Point of Contact 2"],
        ),
        "POC 3": find_column(
            columns,
            ["POC 3", "POC3", "POC-3", "Point of Contact 3"],
        ),
        "Lat": find_column(columns, ["Lat", "Latitude"]),
        "Long": find_column(columns, ["Long", "Longitude", "Lon"]),
        "Check Status": find_column(
            columns,
            ["Check Status", "Status", "Checking Status"],
        ),
    }

    # Priority source order:
    # 1. Priority as per 003 sheet
    # 2. Priority as per Amresh Sir 28.07.2023
    # 3. First column starting with Priority
    priority_col = find_column(
        columns,
        ["Priority as per 003 sheet"],
    )

    if priority_col is None:
        priority_col = find_column(
            columns,
            ["Priority as per Amresh Sir 28.07.2023"],
        )

    if priority_col is None:
        priority_col = find_column(
            columns,
            starts_with=["Priority"],
        )

    result = pd.DataFrame(index=raw.index)

    for target, source in mapping.items():
        result[target] = raw[source] if source else ""

    result["Priority"] = raw[priority_col] if priority_col else ""

    result["Capacity (MW)"] = pd.to_numeric(
        result["Capacity (MW)"]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0)

    text_columns = [
        "PSS", "State", "Priority", "POC1",
        "POC 2", "POC 3", "Lat", "Long", "Check Status",
    ]

    for col in text_columns:
        result[col] = result[col].map(clean)

    result = result[
        result["PSS"].astype(str).str.strip().ne("")
    ].copy()

    return result.reset_index(drop=True)


# =========================================================
# DASHBOARD STYLING
# =========================================================
st.html("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 0% 0%, #e7edff 0%, transparent 32%),
        radial-gradient(ellipse at 100% 0%, #f0eaff 0%, transparent 29%),
        #f6f8fd;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.dashboard-hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 23px 25px;
    margin-bottom: 18px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(115deg, #243f9d 0%, #4267e8 53%, #8058ed 100%);
    box-shadow: 0 12px 30px rgba(58, 75, 175, .19);
}

.dashboard-hero:after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -55px;
    top: -130px;
    border-radius: 50%;
    background: rgba(255,255,255,.10);
    box-shadow: 0 0 0 27px rgba(255,255,255,.045),
                0 0 0 54px rgba(255,255,255,.025);
}

.hero-logo {
    position: relative;
    z-index: 1;
    width: 55px;
    height: 55px;
    flex: 0 0 55px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255,255,255,.28);
    border-radius: 17px;
    background: rgba(255,255,255,.16);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.15);
}

.hero-logo svg {
    width: 34px;
    height: 34px;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-title {
    color: #fff;
    font-size: 27px;
    line-height: 1.2;
    letter-spacing: -.7px;
    font-weight: 850;
}

.hero-subtitle {
    color: rgba(255,255,255,.79);
    font-size: 12px;
    margin-top: 6px;
}

.hero-tag {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 9px;
    border: 1px solid rgba(255,255,255,.24);
    border-radius: 30px;
    background: rgba(255,255,255,.12);
    color: #fff;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .5px;
}

.section-label {
    color: #68748b;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 20px 0 10px;
}

.metric-card {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 18px;
    min-height: 91px;
    border: 1px solid rgba(255,255,255,.85);
    border-radius: 18px;
    box-shadow: 0 7px 20px rgba(42,58,111,.055);
}

.metric-records {
    background: linear-gradient(120deg, #e8efff 0%, #f7f5ff 100%);
}

.metric-capacity {
    background: linear-gradient(120deg, #ddf8f1 0%, #f0fbff 100%);
}

.metric-icon {
    width: 47px;
    height: 47px;
    flex: 0 0 47px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    background: rgba(255,255,255,.78);
    box-shadow: 0 4px 11px rgba(39,60,120,.07);
}

.metric-icon svg {
    width: 24px;
    height: 24px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.metric-records .metric-icon {
    color: #4169f5;
}

.metric-capacity .metric-icon {
    color: #0b9b83;
}

.metric-label {
    color: #65728c;
    font-size: 10px;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .85px;
}

.metric-value {
    color: #202c49;
    font-size: 25px;
    font-weight: 900;
    line-height: 1.2;
    letter-spacing: -.5px;
    margin-top: 4px;
}

.results-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    color: #768198;
    font-size: 12px;
    margin: 15px 0 8px;
}

.results-line strong {
    color: #26324a;
}

div[data-testid="stButton"] > button {
    min-height: 32px;
    padding: 4px 10px;
    border-radius: 9px;
    font-weight: 750;
    border-color: #e1e7f1;
    background: rgba(255,255,255,.9);
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    color: #59667d;
    font-size: 12px;
    font-weight: 750;
}

.empty-state {
    padding: 34px 15px;
    text-align: center;
    color: #77839a;
    background: white;
    border: 1px dashed #dbe2ee;
    border-radius: 17px;
}

@media (max-width: 700px) {
    .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }

    .dashboard-hero {
        padding: 17px;
        border-radius: 18px;
        gap: 12px;
    }

    .hero-logo {
        width: 45px;
        height: 45px;
        flex-basis: 45px;
    }

    .hero-title {
        font-size: 22px;
    }

    .metric-card {
        padding: 12px;
        gap: 9px;
    }

    .metric-value {
        font-size: 21px;
    }
}
</style>
""")


# =========================================================
# HEADER
# =========================================================
st.html(f"""
<div class="dashboard-hero">
  <div class="hero-logo">{BOLT_SVG}</div>
  <div class="hero-content">
    <div class="hero-title">Intimation Dashboard</div>
    <div class="hero-subtitle">
      Power Substation Intelligence · Records, capacity and contacts
    </div>
    <div class="hero-tag">⚡ PSS MANAGEMENT</div>
  </div>
</div>
""")


# =========================================================
# LOAD DATA
# =========================================================
if not EXCEL_PATH.exists():
    st.error(f"Excel file nahi mili: `{EXCEL_PATH}`")
    st.info(
        "`intimation.xlsx` ko project root mein rakho aur "
        "is code ko `pages/9_Intimation_Dashboard.py` mein save karo."
    )
    st.stop()

try:
    data = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error("Excel file load nahi ho paayi. File aur column headers check karo.")
    st.exception(exc)
    st.stop()

if data.empty:
    st.warning("Excel file mein koi PSS record nahi mila.")
    st.stop()


# =========================================================
# FILTERS
# =========================================================
st.html('<div class="section-label">Find records</div>')

filter_a, filter_b, filter_c = st.columns(
    [1.5, 1, 1.2],
    gap="medium",
)

with filter_a:
    pss_search = st.text_input(
        "Search PSS",
        placeholder="Type PSS name...",
        key="intimation_pss_search",
        on_change=reset_page,
    ).strip()

search_base = data.copy()

if pss_search:
    search_base = search_base[
        search_base["PSS"].str.contains(
            pss_search,
            case=False,
            na=False,
            regex=False,
        )
    ]

poc_options = unique_sorted(search_base["POC1"])
poc_choices = [ALL_USERS] + poc_options

# Reset an old POC selection if it is no longer available.
if (
    "intimation_poc_filter" in st.session_state
    and st.session_state["intimation_poc_filter"] not in poc_choices
):
    st.session_state["intimation_poc_filter"] = ALL_USERS

with filter_b:
    selected_poc = st.selectbox(
        "POC 1",
        options=poc_choices,
        key="intimation_poc_filter",
        on_change=reset_page,
    )

state_base = search_base.copy()

if selected_poc != ALL_USERS:
    state_base = state_base[
        state_base["POC1"] == selected_poc
    ]

state_options = unique_sorted(state_base["State"])

# Keep only state selections valid for the current PSS and POC filters.
if "intimation_state_filter" in st.session_state:
    st.session_state["intimation_state_filter"] = [
        state
        for state in st.session_state["intimation_state_filter"]
        if state in state_options
    ]

with filter_c:
    selected_states = st.multiselect(
        "State",
        options=state_options,
        default=[],
        placeholder="All states",
        key="intimation_state_filter",
        on_change=reset_page,
    )

filtered = search_base.copy()

if selected_poc != ALL_USERS:
    filtered = filtered[
        filtered["POC1"] == selected_poc
    ]

if selected_states:
    filtered = filtered[
        filtered["State"].isin(selected_states)
    ]

filtered = filtered.reset_index(drop=True)


# =========================================================
# OVERVIEW METRIC ICONS
# =========================================================
def metric_svg(kind):
    if kind == "records":
        return """
        <svg viewBox="0 0 24 24">
          <rect x="4" y="3.5" width="16" height="17" rx="3"></rect>
          <path d="M8 8h8M8 12h8M8 16h5"></path>
        </svg>
        """

    return """
    <svg viewBox="0 0 24 24">
      <path d="M13.2 2.8 5.8 13h5.1l-.4 8.2L18.2 10h-5.1l.1-7.2Z"></path>
    </svg>
    """


# =========================================================
# OVERVIEW — ONLY TWO CARDS
# =========================================================
total_records = len(filtered)
total_capacity = float(filtered["Capacity (MW)"].sum())

st.html('<div class="section-label">Overview</div>')

metric1, metric2 = st.columns(2, gap="medium")

with metric1:
    st.html(f"""
    <div class="metric-card metric-records">
      <div class="metric-icon">{metric_svg("records")}</div>
      <div>
        <div class="metric-label">Matching Records</div>
        <div class="metric-value">{total_records:,}</div>
      </div>
    </div>
    """)

with metric2:
    st.html(f"""
    <div class="metric-card metric-capacity">
      <div class="metric-icon">{metric_svg("capacity")}</div>
      <div>
        <div class="metric-label">Total Capacity</div>
        <div class="metric-value">
          {format_capacity(total_capacity)}
          <span style="font-size:12px;color:#718198;font-weight:750">MW</span>
        </div>
      </div>
    </div>
    """)


# =========================================================
# PAGINATION
# =========================================================
st.html(f"""
<div class="results-line">
  <span><strong>{total_records:,}</strong> PSS records found</span>
  <span>Click a card to expand · ⚡ to copy PSS name</span>
</div>
""")

if "intimation_page" not in st.session_state:
    st.session_state["intimation_page"] = 0

max_pages = max(
    1,
    (total_records + PAGE_SIZE - 1) // PAGE_SIZE,
)

st.session_state["intimation_page"] = max(
    0,
    min(st.session_state["intimation_page"], max_pages - 1),
)

nav_left, nav_mid, nav_right = st.columns([1, 3, 1])

with nav_left:
    if st.button(
        "← Previous",
        disabled=st.session_state["intimation_page"] <= 0,
        use_container_width=True,
    ):
        st.session_state["intimation_page"] -= 1
        st.rerun()

with nav_mid:
    st.markdown(
        f"""
        <div style="text-align:center;padding-top:7px;
                    color:#78849a;font-size:12px">
          Page <b style="color:#26324a">
          {st.session_state['intimation_page'] + 1}</b>
          of <b style="color:#26324a">{max_pages}</b>
          · {PAGE_SIZE} records per page
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    if st.button(
        "Next →",
        disabled=st.session_state["intimation_page"] >= max_pages - 1,
        use_container_width=True,
    ):
        st.session_state["intimation_page"] += 1
        st.rerun()

page_start = st.session_state["intimation_page"] * PAGE_SIZE
page_df = filtered.iloc[page_start:page_start + PAGE_SIZE]


# =========================================================
# CARD RENDERER
# =========================================================
def render_pss_card(record, card_key):
    pss_name = clean(record["PSS"])
    state_name = clean(record["State"]) or "—"
    poc1 = clean(record["POC1"]) or "—"
    capacity = format_capacity(record["Capacity (MW)"])

    details = {
        "Priority": clean(record["Priority"]) or "—",
        "POC 2": clean(record["POC 2"]) or "—",
        "POC 3": clean(record["POC 3"]) or "—",
        "Latitude": clean(record["Lat"]) or "—",
        "Longitude": clean(record["Long"]) or "—",
        "Check Status": clean(record["Check Status"]) or "—",
    }

    details_html = ""

    for label, value in details.items():
        if label == "Check Status":
            status_class = status_style(value)
            value_html = (
                f'<span class="status-pill {status_class}">'
                f'{safe_html(value)}</span>'
            )
        else:
            value_html = safe_html(value)

        details_html += f"""
        <div class="detail-item">
          <div class="detail-label">{safe_html(label)}</div>
          <div class="detail-value">{value_html}</div>
        </div>
        """

    pss_json = json.dumps(pss_name, ensure_ascii=False)

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{ box-sizing: border-box; }}

        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .pss-card {{
          padding: 11px 12px 10px;
          margin: 0 0 10px;
          border: 1px solid #e5eaf4;
          border-radius: 15px;
          background: linear-gradient(145deg, #ffffff, #fcfdff);
          box-shadow: 0 4px 13px rgba(28,42,78,.04);
          cursor: pointer;
          transition: border-color .17s ease, box-shadow .17s ease,
                      transform .17s ease;
          outline: none;
        }}

        .pss-card:hover {{
          border-color: #c7d3ff;
          box-shadow: 0 8px 20px rgba(42,65,130,.085);
          transform: translateY(-1px);
        }}

        .pss-card:focus-visible {{
          border-color: #6681ff;
          box-shadow: 0 0 0 3px rgba(88,115,255,.14);
        }}

        .card-top {{
          display: flex;
          align-items: center;
          gap: 8px;
        }}

        .card-heading {{
          flex: 1;
          min-width: 0;
        }}

        .pss-name {{
          color: #202b43;
          font-size: 14px;
          font-weight: 850;
          line-height: 1.3;
          overflow-wrap: anywhere;
        }}

        .pss-caption {{
          color: #9aa3b4;
          font-size: 9px;
          letter-spacing: .55px;
          margin-top: 2px;
        }}

        .bolt-copy {{
          width: 34px;
          height: 34px;
          flex: 0 0 34px;
          padding: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 1px solid #dfe3ff;
          border-radius: 11px;
          background: linear-gradient(145deg, #edf3ff, #f0eaff);
          box-shadow: 0 3px 8px rgba(78,91,202,.10);
          cursor: pointer;
          transition: transform .15s ease, box-shadow .15s ease;
        }}

        .bolt-copy:hover {{
          transform: translateY(-1px) scale(1.05);
          box-shadow: 0 5px 12px rgba(78,91,202,.18);
        }}

        .bolt-copy svg {{ width: 22px; height: 22px; }}

        .card-stats {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 7px;
          margin-top: 9px;
        }}

        .card-stat {{
          min-width: 0;
          padding: 6px 8px;
          border: 1px solid #edf0f7;
          border-radius: 9px;
          background: #f7f9fd;
        }}

        .stat-label {{
          color: #8a94a8;
          font-size: 9px;
          font-weight: 850;
          letter-spacing: .65px;
          text-transform: uppercase;
        }}

        .stat-value {{
          color: #34415a;
          font-size: 12px;
          font-weight: 750;
          line-height: 1.35;
          margin-top: 2px;
          overflow-wrap: anywhere;
        }}

        .poc-row {{
          display: flex;
          align-items: baseline;
          gap: 7px;
          margin-top: 7px;
        }}

        .poc-label {{
          flex: 0 0 auto;
          color: #929caf;
          font-size: 10px;
          font-weight: 800;
        }}

        .poc-value {{
          color: #4b5870;
          font-size: 11px;
          font-weight: 650;
          overflow-wrap: anywhere;
        }}

        .expand-hint {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 8px;
          padding-top: 7px;
          border-top: 1px solid #f0f2f8;
          color: #9aa3b5;
          font-size: 9px;
          font-weight: 750;
        }}

        .chevron {{
          display: inline-block;
          color: #7688d9;
          font-size: 13px;
          transition: transform .18s ease;
        }}

        .pss-card.expanded .chevron {{
          transform: rotate(180deg);
        }}

        .detail-panel {{
          display: none;
          grid-template-columns: 1fr 1fr;
          gap: 10px 12px;
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid #e9edf6;
        }}

        .pss-card.expanded .detail-panel {{
          display: grid;
        }}

        .detail-item {{ min-width: 0; }}

        .detail-label {{
          color: #929caf;
          font-size: 9px;
          font-weight: 850;
          letter-spacing: .6px;
          text-transform: uppercase;
        }}

        .detail-value {{
          color: #3b4760;
          font-size: 11px;
          font-weight: 650;
          margin-top: 3px;
          overflow-wrap: anywhere;
        }}

        .status-pill {{
          display: inline-block;
          padding: 3px 7px;
          border-radius: 7px;
          font-size: 10px;
          font-weight: 750;
        }}

        .status-good {{ background: #e2f8ee; color: #11825a; }}
        .status-pending {{ background: #fff0db; color: #a96308; }}
        .status-neutral {{ background: #edf1f8; color: #5e6a80; }}

        .toast {{
          position: fixed;
          right: 9px;
          top: 8px;
          z-index: 10;
          max-width: calc(100% - 18px);
          padding: 7px 10px;
          border-radius: 9px;
          color: #fff;
          background: #202b43;
          font-size: 11px;
          font-weight: 750;
          box-shadow: 0 5px 18px #17203325;
          opacity: 0;
          transform: translateY(-4px);
          transition: all .18s ease;
          pointer-events: none;
          overflow-wrap: anywhere;
        }}

        .toast.show {{
          opacity: 1;
          transform: translateY(0);
        }}
      </style>
    </head>

    <body>
      <div class="pss-card"
           id="pssCard"
           role="button"
           tabindex="0"
           aria-expanded="false"
           title="Click to expand or collapse details">

        <div class="card-top">
          <div class="card-heading">
            <div class="pss-name">{safe_html(pss_name)}</div>
            <div class="pss-caption">POWER SUBSTATION</div>
          </div>

          <button class="bolt-copy"
                  type="button"
                  title="Copy PSS name"
                  aria-label="Copy PSS name">
            {BOLT_SVG}
          </button>
        </div>

        <div class="card-stats">
          <div class="card-stat">
            <div class="stat-label">Capacity</div>
            <div class="stat-value">{safe_html(capacity)} MW</div>
          </div>
          <div class="card-stat">
            <div class="stat-label">State</div>
            <div class="stat-value">{safe_html(state_name)}</div>
          </div>
        </div>

        <div class="poc-row">
          <span class="poc-label">POC 1</span>
          <span class="poc-value">{safe_html(poc1)}</span>
        </div>

        <div class="expand-hint">
          <span class="hint-text">Click card to view details</span>
          <span class="chevron">⌄</span>
        </div>

        <div class="detail-panel">
          {details_html}
        </div>
      </div>

      <div class="toast" id="copyToast" role="status">
        ✓ PSS name copied
      </div>

      <script>
        const card = document.getElementById("pssCard");
        const copyButton = card.querySelector(".bolt-copy");
        const hint = card.querySelector(".hint-text");
        const toast = document.getElementById("copyToast");
        let toastTimer;

        function updateFrameHeight() {{
          // Ask the Streamlit iframe to fit the card's current height.
          const height = Math.ceil(
            document.documentElement.scrollHeight
          ) + 2;

          window.parent.postMessage(
            {{
              isStreamlitMessage: true,
              type: "streamlit:setFrameHeight",
              height: height
            }},
            "*"
          );
        }}

        function toggleCard() {{
          const expanded = card.classList.toggle("expanded");

          card.setAttribute("aria-expanded", String(expanded));
          hint.textContent = expanded
            ? "Click card to hide details"
            : "Click card to view details";

          requestAnimationFrame(updateFrameHeight);
          setTimeout(updateFrameHeight, 100);
        }}

        card.addEventListener("click", function(event) {{
          if (event.target.closest(".bolt-copy")) return;
          toggleCard();
        }});

        card.addEventListener("keydown", function(event) {{
          if (event.target !== card) return;

          if (event.key === "Enter" || event.key === " ") {{
            event.preventDefault();
            toggleCard();
          }}
        }});

        copyButton.addEventListener("click", async function(event) {{
          event.stopPropagation();

          const name = {pss_json};

          try {{
            if (navigator.clipboard && window.isSecureContext) {{
              await navigator.clipboard.writeText(name);
            }} else {{
              const textarea = document.createElement("textarea");
              textarea.value = name;
              textarea.style.position = "fixed";
              textarea.style.left = "-9999px";
              document.body.appendChild(textarea);
              textarea.focus();
              textarea.select();

              const copied = document.execCommand("copy");
              textarea.remove();

              if (!copied) throw new Error("Clipboard unavailable");
            }}

            toast.textContent = "✓ Copied: " + name;
            toast.classList.add("show");

            copyButton.style.background =
              "linear-gradient(145deg,#dffaf0,#e8f5ff)";

            clearTimeout(toastTimer);

            toastTimer = setTimeout(function() {{
              toast.classList.remove("show");
              copyButton.style.background =
                "linear-gradient(145deg,#edf3ff,#f0eaff)";
            }}, 1600);

          }} catch (error) {{
            toast.textContent = "Browser clipboard access unavailable";
            toast.classList.add("show");

            clearTimeout(toastTimer);

            toastTimer = setTimeout(function() {{
              toast.classList.remove("show");
            }}, 1800);
          }}
        }});

        window.addEventListener("load", function() {{
          updateFrameHeight();
          setTimeout(updateFrameHeight, 150);
        }});
      </script>
    </body>
    </html>
    """

    # The card expands within its iframe and requests a matching height.
    # This keeps collapsed cards compact without a separate details button.
    components.html(
        html,
        height=155,
        scrolling=False
    )


# =========================================================
# DISPLAY CARDS — 2 COLUMNS, 20 RECORDS PER PAGE
# =========================================================
if page_df.empty:
    st.html("""
    <div class="empty-state">
      <strong>No records found</strong><br>
      Search ya filters change karke dekho.
    </div>
    """)
else:
    for pair_start in range(0, len(page_df), 2):
        card_columns = st.columns(2, gap="medium")
        pair = page_df.iloc[pair_start:pair_start + 2]

        for offset, (_, record) in enumerate(pair.iterrows()):
            row_number = page_start + pair_start + offset

            # Stable key for this displayed row.
            card_key = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                f"{row_number}_{clean(record['PSS'])}",
            )[:90]

            with card_columns[offset]:
                render_pss_card(record, card_key)


# =========================================================
# FOOTER
# =========================================================
st.caption(
    "Tip: Card par click karke details expand/collapse karo. "
    "⚡ sticker par click karke PSS name copy karo."
)
