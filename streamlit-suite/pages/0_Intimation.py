
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
def normalize_header(value):
    value = str(value).strip().lower()
    value = re.sub(r"[\n\r\t]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def find_column(columns, candidates=(), starts_with=()):
    normalized = {column: normalize_header(column) for column in columns}

    for candidate in candidates:
        target = normalize_header(candidate)
        for column, value in normalized.items():
            if value == target:
                return column

    for prefix in starts_with:
        target = normalize_header(prefix)
        for column, value in normalized.items():
            if value.startswith(target):
                return column

    return None


def clean(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def safe_html(value):
    return escape(clean(value), quote=True)


def unique_sorted(series):
    values = [clean(value) for value in series.tolist()]
    return sorted(
        {value for value in values if value},
        key=str.casefold,
    )


def format_capacity(value):
    try:
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
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
      <stop offset="0%" stop-color="#fff3a6"/>
      <stop offset="48%" stop-color="#ffc83d"/>
      <stop offset="100%" stop-color="#ff8a22"/>
    </linearGradient>
  </defs>
  <path d="M17.9 2.7 7 17.1h7l-1.1 12.2L25 13.7h-7.3l.2-11Z"
        fill="url(#boltFill)"
        stroke="#8c83ff"
        stroke-width="1.1"
        stroke-linejoin="round"/>
</svg>
"""


# =========================================================
# LOAD EXCEL
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(file_path, modified_time):
    # modified_time refreshes the cache when the Excel file changes.
    del modified_time

    raw = pd.read_excel(file_path, engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
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

    # Priority source preference:
    # 1. Priority as per 003 sheet
    # 2. Priority as per Amresh Sir 28.07.2023
    # 3. First column beginning with Priority
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
        if source is not None:
            result[target] = raw[source]
        else:
            result[target] = ""

    if priority_col is not None:
        result["Priority"] = raw[priority_col]
    else:
        result["Priority"] = ""

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

    for column in text_columns:
        result[column] = result[column].map(clean)

    result = result[
        result["PSS"].astype(str).str.strip().ne("")
    ].copy()

    return result.reset_index(drop=True)


# =========================================================
# CSS
# =========================================================
st.html("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, rgba(91,130,255,.08), transparent 34%),
        radial-gradient(ellipse at 96% 3%, rgba(158,104,255,.07), transparent 30%),
        #f7f9fd;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.dashboard-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 5px;
}

.dashboard-logo {
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    border-radius: 15px;
    background: linear-gradient(135deg, #3767f5, #8057f4);
    box-shadow: 0 7px 19px rgba(70,93,232,.23);
}

.dashboard-logo svg {
    width: 28px;
    height: 28px;
}

.dashboard-title {
    font-size: 26px;
    font-weight: 850;
    letter-spacing: -.7px;
    color: #182238;
    line-height: 1.2;
}

.dashboard-subtitle {
    color: #7a8599;
    font-size: 12px;
    margin-top: 4px;
}

.section-label {
    color: #69758b;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 20px 0 9px;
}

.metric-card {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 13px 16px;
    min-height: 80px;
    background: rgba(255,255,255,.95);
    border: 1px solid #e8ecf5;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(27,43,83,.035);
}

.metric-icon {
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    border-radius: 13px;
}

.metric-icon svg {
    width: 23px;
    height: 23px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.metric-icon.records {
    color: #4169f5;
    background: linear-gradient(135deg, #e8efff, #f0edff);
}

.metric-icon.capacity {
    color: #0e9f8b;
    background: linear-gradient(135deg, #dcfaf2, #e9f8ff);
}

.metric-label {
    color: #7b8599;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.metric-value {
    color: #182238;
    font-size: 23px;
    line-height: 1.2;
    font-weight: 850;
    letter-spacing: -.5px;
    margin-top: 3px;
}

.results-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin: 15px 0 8px;
    color: #69758b;
    font-size: 12px;
}

.results-line strong {
    color: #26324a;
}

div[data-testid="stButton"] > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 32px;
    padding: 4px 10px;
}

div[data-testid="stButton"] > button[kind="secondary"] {
    border: 1px solid #e1e7f1;
    background: #fff;
    color: #4e5b73;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stMultiSelect"] label {
    color: #59667d;
    font-size: 12px;
    font-weight: 750;
}

.empty-state {
    text-align: center;
    padding: 30px 15px;
    color: #7c879a;
    background: #fff;
    border: 1px dashed #dbe2ee;
    border-radius: 15px;
}

@media (max-width: 700px) {
    .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }

    .dashboard-title {
        font-size: 22px;
    }

    .metric-card {
        padding: 11px;
        gap: 9px;
    }

    .metric-value {
        font-size: 20px;
    }
}
</style>
""")


# =========================================================
# HEADER
# =========================================================
st.html(f"""
<div class="dashboard-header">
  <div class="dashboard-logo">{BOLT_SVG}</div>
  <div>
    <div class="dashboard-title">Intimation Dashboard</div>
    <div class="dashboard-subtitle">
      Power Substation records · Overview and contact details
    </div>
  </div>
</div>
""")


# =========================================================
# READ DATA
# =========================================================
if not EXCEL_PATH.exists():
    st.error(f"Excel file nahi mili: `{EXCEL_PATH}`")
    st.info(
        "Ensure `intimation.xlsx` project root mein ho aur "
        "dashboard `pages/9_Intimation_Dashboard.py` mein ho."
    )
    st.stop()

try:
    data = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error("Excel file load nahi ho paayi. File aur column headers check karein.")
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

# Reset invalid selections when search results change.
if selected_poc not in poc_choices:
    st.session_state["intimation_poc_filter"] = ALL_USERS
    selected_poc = ALL_USERS

valid_states = [
    state for state in selected_states
    if state in state_options
]

if valid_states != selected_states:
    st.session_state["intimation_state_filter"] = valid_states
    selected_states = valid_states


# =========================================================
# OVERVIEW — ONLY TWO METRICS
# =========================================================
def metric_svg(kind):
    if kind == "records":
        return """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="4" y="3.5" width="16" height="17" rx="3"></rect>
          <path d="M8 8h8M8 12h8M8 16h5"></path>
        </svg>
        """

    return """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M13.2 2.8 5.8 13h5.1l-.4 8.2L18.2 10h-5.1l.1-7.2Z"></path>
    </svg>
    """


total_records = len(filtered)
total_capacity = float(filtered["Capacity (MW)"].sum())

st.html('<div class="section-label">Overview</div>')

metric1, metric2 = st.columns(2, gap="medium")

with metric1:
    st.html(f"""
    <div class="metric-card">
      <div class="metric-icon records">{metric_svg("records")}</div>
      <div>
        <div class="metric-label">Matching Records</div>
        <div class="metric-value">{total_records:,}</div>
      </div>
    </div>
    """)

with metric2:
    st.html(f"""
    <div class="metric-card">
      <div class="metric-icon capacity">{metric_svg("capacity")}</div>
      <div>
        <div class="metric-label">Total Capacity</div>
        <div class="metric-value">
          {format_capacity(total_capacity)}
          <span style="font-size:12px;color:#7c879a;font-weight:700">MW</span>
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
  <span>⚡ Click lightning to copy PSS name</span>
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
# COMPACT PSS CARDS
# =========================================================
if page_df.empty:
    st.html("""
    <div class="empty-state">
      <strong>No records found</strong><br>
      Search ya filters change karke dekhein.
    </div>
    """)
else:
    for pair_start in range(0, len(page_df), 2):
        card_columns = st.columns(2, gap="medium")
        pair = page_df.iloc[pair_start:pair_start + 2]

        for offset, (_, record) in enumerate(pair.iterrows()):
            pss_name = clean(record["PSS"])
            state_name = clean(record["State"]) or "—"
            poc1 = clean(record["POC1"]) or "—"
            capacity = format_capacity(record["Capacity (MW)"])

            priority = clean(record["Priority"]) or "—"
            poc2 = clean(record["POC 2"]) or "—"
            poc3 = clean(record["POC 3"]) or "—"
            latitude = clean(record["Lat"]) or "—"
            longitude = clean(record["Long"]) or "—"
            check_status = clean(record["Check Status"]) or "—"

            row_number = page_start + pair_start + offset
            unique_key = f"{row_number}_{pss_name}_{state_name}_{poc1}"
            unique_key = re.sub(r"[^a-zA-Z0-9_-]", "_", unique_key)[:100]

            expanded_key = f"intimation_expanded_{unique_key}"

            if expanded_key not in st.session_state:
                st.session_state[expanded_key] = False

            expanded = st.session_state[expanded_key]
            status_class = status_style(check_status)

            details_html = ""

            if expanded:
                details_html = f"""
                <div class="detail-panel">
                  <div class="detail-item">
                    <div class="detail-label">Priority</div>
                    <div class="detail-value">{safe_html(priority)}</div>
                  </div>
                  <div class="detail-item">
                    <div class="detail-label">POC 2</div>
                    <div class="detail-value">{safe_html(poc2)}</div>
                  </div>
                  <div class="detail-item">
                    <div class="detail-label">POC 3</div>
                    <div class="detail-value">{safe_html(poc3)}</div>
                  </div>
                  <div class="detail-item">
                    <div class="detail-label">Latitude</div>
                    <div class="detail-value">{safe_html(latitude)}</div>
                  </div>
                  <div class="detail-item">
                    <div class="detail-label">Longitude</div>
                    <div class="detail-value">{safe_html(longitude)}</div>
                  </div>
                  <div class="detail-item">
                    <div class="detail-label">Check Status</div>
                    <div class="detail-value">
                      <span class="status-pill {status_class}">
                        {safe_html(check_status)}
                      </span>
                    </div>
                  </div>
                </div>
                """

            # JSON escaping protects PSS names containing quotes or symbols.
            pss_json = json.dumps(pss_name, ensure_ascii=False)

            card_html = f"""
            <div class="compact-card">
              <div class="card-top">
                <div class="card-heading">
                  <div class="pss-name">{safe_html(pss_name)}</div>
                  <div class="pss-caption">POWER SUBSTATION</div>
                </div>

                <button class="bolt-copy"
                        type="button"
                        title="Copy PSS name"
                        aria-label="Copy PSS name"
                        onclick='copyPssName(this, {pss_json})'>
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

              {details_html}
            </div>
            """

            # Isolated iframe makes the lightning sticker clickable and
            # enables browser clipboard access without a separate copy button.
            with card_columns[offset]:
                components.html(
                    f"""
                    <!doctype html>
                    <html>
                    <head>
                      <meta charset="utf-8">
                      <style>
                        * {{ box-sizing: border-box; }}
                        html, body {{
                          margin: 0;
                          padding: 0;
                          font-family: -apple-system, BlinkMacSystemFont,
                                       "Segoe UI", sans-serif;
                          background: transparent;
                        }}
                        .compact-card {{
                          border: 1px solid #e7ebf4;
                          border-radius: 15px;
                          background: #fff;
                          box-shadow: 0 4px 14px rgba(28,42,78,.035);
                          padding: 11px 12px 10px;
                          margin: 0;
                          transition: border-color .16s ease,
                                      box-shadow .16s ease;
                        }}
                        .compact-card:hover {{
                          border-color: #cdd7ff;
                          box-shadow: 0 7px 18px rgba(42,65,130,.07);
                        }}
                        .card-top {{
                          display: flex;
                          align-items: center;
                          gap: 8px;
                          min-width: 0;
                        }}
                        .card-heading {{ min-width: 0; flex: 1; }}
                        .pss-name {{
                          font-size: 14px;
                          font-weight: 800;
                          color: #202b43;
                          line-height: 1.3;
                          overflow-wrap: anywhere;
                        }}
                        .pss-caption {{
                          font-size: 9px;
                          color: #9aa3b4;
                          margin-top: 2px;
                          letter-spacing: .5px;
                        }}
                        .bolt-copy {{
                          width: 33px;
                          height: 33px;
                          flex: 0 0 33px;
                          border: 1px solid rgba(113,133,255,.23);
                          border-radius: 11px;
                          padding: 0;
                          display: flex;
                          align-items: center;
                          justify-content: center;
                          background: linear-gradient(145deg,#edf3ff,#f0eaff);
                          box-shadow: 0 3px 8px rgba(78,91,202,.10);
                          cursor: pointer;
                          transition: transform .15s ease,
                                      box-shadow .15s ease;
                        }}
                        .bolt-copy:hover {{
                          transform: translateY(-1px) scale(1.04);
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
                          background: #f7f9fd;
                          border: 1px solid #eef1f7;
                          border-radius: 9px;
                          padding: 6px 8px;
                        }}
                        .stat-label {{
                          font-size: 9px;
                          color: #8a94a8;
                          text-transform: uppercase;
                          letter-spacing: .65px;
                          font-weight: 800;
                        }}
                        .stat-value {{
                          font-size: 12px;
                          font-weight: 750;
                          color: #34415a;
                          margin-top: 2px;
                          overflow-wrap: anywhere;
                          line-height: 1.35;
                        }}
                        .poc-row {{
                          margin-top: 7px;
                          display: flex;
                          align-items: baseline;
                          gap: 7px;
                          min-width: 0;
                        }}
                        .poc-label {{
                          flex: 0 0 auto;
                          font-size: 10px;
                          color: #929caf;
                          font-weight: 750;
                        }}
                        .poc-value {{
                          font-size: 11px;
                          color: #4b5870;
                          font-weight: 650;
                          overflow-wrap: anywhere;
                        }}
                        .detail-panel {{
                          border-top: 1px solid #edf0f6;
                          margin-top: 9px;
                          padding-top: 9px;
                          display: grid;
                          grid-template-columns: 1fr 1fr;
                          gap: 9px 12px;
                        }}
                        .detail-item {{ min-width: 0; }}
                        .detail-label {{
                          font-size: 9px;
                          text-transform: uppercase;
                          letter-spacing: .65px;
                          color: #929caf;
                          font-weight: 800;
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
                          right: 8px;
                          top: 7px;
                          z-index: 10;
                          padding: 7px 10px;
                          border-radius: 9px;
                          color: white;
                          background: #202b43;
                          font-size: 11px;
                          font-weight: 700;
                          box-shadow: 0 5px 18px #17203325;
                          opacity: 0;
                          transform: translateY(-4px);
                          transition: all .18s ease;
                          pointer-events: none;
                        }}
                        .toast.show {{
                          opacity: 1;
                          transform: translateY(0);
                        }}
                      </style>
                    </head>

                    <body>
                      {card_html}

                      <div class="toast" id="copyToast" role="status">
                        ✓ PSS name copied
                      </div>

                      <script>
                        async function copyPssName(button, name) {{
                          try {{
                            if (navigator.clipboard && window.isSecureContext) {{
                              await navigator.clipboard.writeText(name);
                            }} else {{
                              const input = document.createElement("textarea");
                              input.value = name;
                              input.style.position = "fixed";
                              input.style.opacity = "0";
                              document.body.appendChild(input);
                              input.focus();
                              input.select();

                              const ok = document.execCommand("copy");
                              input.remove();

                              if (!ok) throw new Error("Copy failed");
                            }}

                            const toast = document.getElementById("copyToast");
                            toast.textContent = "✓ Copied: " + name;
                            toast.classList.add("show");

                            button.style.background =
                              "linear-gradient(145deg,#dffaf0,#e8f5ff)";

                            setTimeout(() => {{
                              toast.classList.remove("show");
                              button.style.background =
                                "linear-gradient(145deg,#edf3ff,#f0eaff)";
                            }}, 1500);

                          }} catch (error) {{
                            const toast = document.getElementById("copyToast");
                            toast.textContent = "Copy not supported by browser";
                            toast.classList.add("show");

                            setTimeout(() => {{
                              toast.classList.remove("show");
                            }}, 1800);
                          }}
                        }}
                      </script>
                    </body>
                    </html>
                    """,
                    height=(
                        205 + 125
                        if expanded
                        else 160
                    ),
                    scrolling=False,
                )

                if st.button(
                    "⌄  Show details" if not expanded else "⌃  Hide details",
                    key=f"toggle_{unique_key}",
                    use_container_width=True,
                ):
                    st.session_state[expanded_key] = not expanded
                    st.rerun()


st.caption(
    "Tip: ⚡ sticker par click karke PSS name copy karein. "
    "Excel update karne ke baad dashboard refresh karein."
)
