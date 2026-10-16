
import html
import json
from pathlib import Path

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
)


# ============================================================
# STYLING
# ============================================================

st.html("""
<style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    .dashboard-header {
        background: linear-gradient(120deg, #123c69 0%, #176b87 55%, #19a7a0 100%);
        color: white;
        padding: 25px 28px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(18, 60, 105, 0.18);
    }

    .dashboard-header h1 {
        color: white !important;
        font-size: 30px !important;
        font-weight: 800 !important;
        margin: 0 0 7px 0 !important;
    }

    .dashboard-header p {
        color: #e1f5f5 !important;
        margin: 0 !important;
        font-size: 14px;
    }

    .metric-card {
        padding: 20px 22px;
        border-radius: 16px;
        color: white;
        min-height: 112px;
        box-shadow: 0 7px 20px rgba(15, 23, 42, 0.12);
        margin-bottom: 8px;
    }

    .metric-blue {
        background: linear-gradient(135deg, #164e8a, #3182ce);
    }

    .metric-teal {
        background: linear-gradient(135deg, #087f8c, #20b2aa);
    }

    .metric-label {
        font-size: 13px;
        font-weight: 600;
        opacity: 0.92;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-subtitle {
        font-size: 12px;
        margin-top: 7px;
        opacity: 0.88;
    }

    .section-heading {
        font-size: 20px;
        font-weight: 800;
        color: var(--text-color);
        margin: 18px 0 10px 0;
    }

    .pss-name {
        font-size: 16px;
        font-weight: 800;
        overflow-wrap: anywhere;
        padding-top: 5px;
    }

    .pss-summary {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    .detail-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
    }

    .detail-value {
        font-size: 14px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 15px;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 650;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        font-weight: 650;
    }

    .results-caption {
        color: #64748b;
        font-size: 13px;
        padding: 5px 0 12px 0;
    }
</style>
""")


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="dashboard-header">
    <h1>⚡ Intimation Dashboard</h1>
    <p>Power System Sites · Intimation Tracking · PSS Overview</p>
</div>
""")


# ============================================================
# HELPERS
# ============================================================

def normalize_column(value):
    """Normalize column labels for flexible matching."""
    return "".join(
        char.lower()
        for char in str(value).strip()
        if char.isalnum()
    )


def find_column(df, candidates, starts_with=None):
    """
    Find a dataframe column using preferred names first.
    Matching ignores spaces, underscores and punctuation.
    """
    normalized = {
        normalize_column(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column(candidate)
        if key in normalized:
            return normalized[key]

    if starts_with:
        prefix = normalize_column(starts_with)

        for col in df.columns:
            if normalize_column(col).startswith(prefix):
                return col

    return None


def clean_text(value):
    """Convert missing values into readable display text."""
    if pd.isna(value):
        return "—"

    text = str(value).strip()

    if text.lower() in {"nan", "none", "nat"}:
        return "—"

    return text if text else "—"


def numeric_value(value):
    """Safely convert a capacity value to a number."""
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce"
    ).iloc[0]


def display_capacity(value):
    number = numeric_value(value)

    if pd.isna(number):
        return clean_text(value)

    return f"{number:,.2f}".rstrip("0").rstrip(".")


def render_copy_button(pss_name, unique_key):
    """
    Render a small independent copy button.
    This iframe is used only for copying; card expansion is
    handled by Streamlit outside the iframe.
    """
    safe_name = json.dumps(str(pss_name), ensure_ascii=False)

    copy_html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:transparent;">
      <button id="copyBtn"
        style="
          width:100%;
          height:38px;
          border:1px solid #dbeafe;
          border-radius:10px;
          background:#eff6ff;
          color:#1d4ed8;
          font-size:18px;
          font-weight:700;
          cursor:pointer;
        "
        title="Copy PSS name"
        aria-label="Copy PSS name">⚡</button>

      <script>
        const pssName = {safe_name};
        const btn = document.getElementById("copyBtn");

        btn.addEventListener("click", async function(event) {{
          event.preventDefault();
          event.stopPropagation();

          try {{
            await navigator.clipboard.writeText(pssName);
            btn.textContent = "✓";
            btn.style.background = "#dcfce7";
            btn.style.color = "#166534";
          }} catch (error) {{
            const textarea = document.createElement("textarea");
            textarea.value = pssName;
            textarea.style.position = "fixed";
            textarea.style.left = "-9999px";
            document.body.appendChild(textarea);
            textarea.select();

            try {{
              document.execCommand("copy");
              btn.textContent = "✓";
              btn.style.background = "#dcfce7";
              btn.style.color = "#166534";
            }} catch (fallbackError) {{
              btn.textContent = "!";
            }}

            textarea.remove();
          }}

          setTimeout(() => {{
            btn.textContent = "⚡";
            btn.style.background = "#eff6ff";
            btn.style.color = "#1d4ed8";
          }}, 1200);
        }});
      </script>
    </body>
    </html>
    """

    components.html(
        copy_html,
        height=42,
        scrolling=False,
    )


def detail_item(label, value):
    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
            <div class="detail-label">{html.escape(str(label))}</div>
            <div class="detail-value">{html.escape(clean_text(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pss_card(record, card_key, columns):
    """
    Native Streamlit card.

    Important:
    - The expander and details are rendered in the parent page.
    - Expanded cards increase their actual layout height.
    - Other cards automatically move down.
    - The copy button does not control expansion.
    """
    pss_name = clean_text(record.get(columns["PSS"], "—"))

    with st.container(border=True):
        title_col, copy_col = st.columns(
            [5.5, 1],
            vertical_alignment="center",
        )

        with title_col:
            st.markdown(
                f'<div class="pss-name">{html.escape(pss_name)}</div>',
                unsafe_allow_html=True,
            )

            capacity = display_capacity(
                record.get(columns["Capacity"], "—")
            )

            state = clean_text(
                record.get(columns["State"], "—")
            )

            poc1 = clean_text(
                record.get(columns["POC1"], "—")
            )

            st.markdown(
                f"""
                <div class="pss-summary">
                    Capacity: <b>{html.escape(capacity)}</b>
                    &nbsp; · &nbsp;
                    State: <b>{html.escape(state)}</b>
                </div>
                <div class="pss-summary">
                    POC1: <b>{html.escape(poc1)}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            render_copy_button(
                pss_name,
                card_key,
            )

        with st.expander(
            "View site details",
            key=f"details_{card_key}",
        ):
            left, right = st.columns(2)

            with left:
                detail_item(
                    "Priority",
                    record.get(columns["Priority"], "—"),
                )
                detail_item(
                    "POC 2",
                    record.get(columns["POC2"], "—"),
                )
                detail_item(
                    "POC 3",
                    record.get(columns["POC3"], "—"),
                )

            with right:
                detail_item(
                    "Latitude",
                    record.get(columns["Latitude"], "—"),
                )
                detail_item(
                    "Longitude",
                    record.get(columns["Longitude"], "—"),
                )
                detail_item(
                    "Check Status",
                    record.get(columns["Check Status"], "—"),
                )


# ============================================================
# LOAD EXCEL
# ============================================================

DATA_FILE = Path(__file__).resolve().parent.parent / "intimation.xlsx"

if not DATA_FILE.exists():
    st.error(
        f"Excel file not found: {DATA_FILE}\n\n"
        "Please place intimation.xlsx in the project root "
        "alongside the pages folder."
    )
    st.stop()


@st.cache_data(show_spinner="Loading intimation data...")
def load_data(file_path, modified_time):
    return pd.read_excel(file_path)


try:
    df = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Unable to read intimation.xlsx: {exc}")
    st.stop()


if df.empty:
    st.warning("The Excel file contains no records.")
    st.stop()


df.columns = [
    str(col).strip()
    for col in df.columns
]


# ============================================================
# COLUMN MAPPING
# ============================================================

columns = {
    "PSS": find_column(
        df,
        ["PSS", "PSS Name", "Substation", "Site Name"],
    ),
    "Capacity": find_column(
        df,
        ["Capacity (MW)", "Capacity MW", "Capacity", "MW"],
    ),
    "State": find_column(
        df,
        ["State", "State Name"],
    ),
    "POC1": find_column(
        df,
        ["POC1", "POC 1", "POC-1", "POC 1 Name"],
    ),
    "POC2": find_column(
        df,
        ["POC 2", "POC2", "POC-2"],
    ),
    "POC3": find_column(
        df,
        ["POC 3", "POC3", "POC-3"],
    ),
    "Latitude": find_column(
        df,
        ["Lat", "Latitude"],
    ),
    "Longitude": find_column(
        df,
        ["Long", "Longitude", "Lon"],
    ),
    "Check Status": find_column(
        df,
        ["Check Status", "Status", "Intimation Status"],
    ),
}

# Preserve the requested priority preference.
columns["Priority"] = find_column(
    df,
    [
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ],
    starts_with="Priority",
)


# Check the essential columns.
required_columns = [
    "PSS",
    "Capacity",
    "State",
    "POC1",
]

missing_columns = [
    name
    for name in required_columns
    if columns.get(name) is None
]

if missing_columns:
    st.error(
        "Required columns could not be identified: "
        + ", ".join(missing_columns)
    )
    st.write("Available Excel columns:", list(df.columns))
    st.stop()


# Optional columns receive a blank placeholder.
for name in [
    "Priority",
    "POC2",
    "POC3",
    "Latitude",
    "Longitude",
    "Check Status",
]:
    if columns.get(name) is None:
        df[f"__missing_{name}"] = "—"
        columns[name] = f"__missing_{name}"


# Remove fully empty PSS rows.
df = df[
    df[columns["PSS"]].notna()
].copy()

if df.empty:
    st.warning("No records with a PSS name were found.")
    st.stop()


# ============================================================
# FILTERS
# ============================================================

st.markdown(
    '<div class="section-heading">🔎 Search & Filters</div>',
    unsafe_allow_html=True,
)

search_col, poc_col, state_col = st.columns(
    [2, 1, 1],
)

with search_col:
    pss_search = st.text_input(
        "Search PSS",
        placeholder="Enter PSS or site name...",
        key="intimation_pss_search",
    )

# Search is applied first.
search_df = df.copy()

if pss_search.strip():
    search_df = search_df[
        search_df[columns["PSS"]]
        .astype(str)
        .str.contains(
            pss_search.strip(),
            case=False,
            na=False,
        )
    ]

# POC1 options depend on the current PSS search.
poc_options = sorted(
    search_df[columns["POC1"]]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda s: s.ne("")]
    .unique()
    .tolist(),
    key=str.casefold,
)

with poc_col:
    selected_poc = st.selectbox(
        "POC1",
        options=["All"] + poc_options,
        key="intimation_poc_filter",
    )

# State options depend on PSS search and selected POC1.
state_source = search_df.copy()

if selected_poc != "All":
    state_source = state_source[
        state_source[columns["POC1"]]
        .astype(str)
        .str.strip()
        .eq(selected_poc)
    ]

state_options = sorted(
    state_source[columns["State"]]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda s: s.ne("")]
    .unique()
    .tolist(),
    key=str.casefold,
)

with state_col:
    selected_state = st.selectbox(
        "State",
        options=["All"] + state_options,
        key="intimation_state_filter",
    )


# Apply all filters.
filtered_df = search_df.copy()

if selected_poc != "All":
    filtered_df = filtered_df[
        filtered_df[columns["POC1"]]
        .astype(str)
        .str.strip()
        .eq(selected_poc)
    ]

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df[columns["State"]]
        .astype(str)
        .str.strip()
        .eq(selected_state)
    ]


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_records = len(filtered_df)

capacity_series = pd.to_numeric(
    filtered_df[columns["Capacity"]]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("MW", "", case=False, regex=False)
    .str.strip(),
    errors="coerce",
)

total_capacity = capacity_series.sum()

if pd.isna(total_capacity):
    total_capacity = 0.0


st.markdown(
    '<div class="section-heading">📊 Overview</div>',
    unsafe_allow_html=True,
)

metric1, metric2 = st.columns(2)

with metric1:
    st.html(f"""
    <div class="metric-card metric-blue">
        <div class="metric-label">MATCHING RECORDS</div>
        <div class="metric-value">{total_records:,}</div>
        <div class="metric-subtitle">PSS records matching the selected filters</div>
    </div>
    """)

with metric2:
    st.html(f"""
    <div class="metric-card metric-teal">
        <div class="metric-label">TOTAL CAPACITY</div>
        <div class="metric-value">{total_capacity:,.2f} MW</div>
        <div class="metric-subtitle">Combined capacity of matching records</div>
    </div>
    """)


# ============================================================
# PAGINATION
# ============================================================

st.markdown(
    '<div class="section-heading">⚡ PSS Sites</div>',
    unsafe_allow_html=True,
)

PAGE_SIZE = 20

total_pages = max(
    1,
    (total_records + PAGE_SIZE - 1) // PAGE_SIZE,
)

if "intimation_page" not in st.session_state:
    st.session_state.intimation_page = 1

# Keep the current page valid when filters change.
st.session_state.intimation_page = min(
    max(1, st.session_state.intimation_page),
    total_pages,
)

page_col, info_col = st.columns(
    [1, 3],
    vertical_alignment="center",
)

with page_col:
    current_page = st.selectbox(
        "Page",
        options=list(range(1, total_pages + 1)),
        index=st.session_state.intimation_page - 1,
        format_func=lambda page: f"Page {page} of {total_pages}",
        key="intimation_page_selector",
    )

st.session_state.intimation_page = current_page

start_index = (current_page - 1) * PAGE_SIZE
end_index = min(start_index + PAGE_SIZE, total_records)

with info_col:
    st.markdown(
        f"""
        <div class="results-caption">
            Showing <b>{start_index + 1 if total_records else 0}
            – {end_index}</b> of <b>{total_records:,}</b> matching records
            · 20 records per page
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RENDER CARDS
# ============================================================

if filtered_df.empty:
    st.info(
        "No matching PSS records found. Try changing the search or filters."
    )
else:
    page_df = filtered_df.iloc[
        start_index:end_index
    ]

    records = page_df.to_dict(orient="records")

    # Render two cards per row. Each card expands in the actual
    # Streamlit layout, so following rows shift down naturally.
    for row_start in range(0, len(records), 2):
        card_cols = st.columns(2, gap="medium")

        with card_cols[0]:
            record = records[row_start]
            render_pss_card(
                record,
                card_key=f"{current_page}_{row_start}",
                columns=columns,
            )

        if row_start + 1 < len(records):
            with card_cols[1]:
                record = records[row_start + 1]
                render_pss_card(
                    record,
                    card_key=f"{current_page}_{row_start + 1}",
                    columns=columns,
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intimation Dashboard · Data source: intimation.xlsx"
)
