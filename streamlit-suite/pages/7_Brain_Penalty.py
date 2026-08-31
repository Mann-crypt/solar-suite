import streamlit as st
import html
import re
from datetime import date

st.set_page_config(
    page_title="Solar PSS Coloring & 10-Sheet Collage",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Configuration
# -----------------------------
N_SHEETS = 10
N_PSS = 31

PALETTE = {
    "Yellow": "#FFF200",
    "Green": "#28B463",
    "Blue": "#2E86DE",
    "Red": "#F4511E",
    "Orange": "#FF9800",
    "Purple": "#9B59B6",
    "Pink": "#EC407A",
    "Cyan": "#19C3C8",
    "White": "#FFFFFF",
}

# Approximate geometry based on the supplied paper layout.
# Each tuple is: pss_number, x, y, width, height.
# The visual page is built as SVG so the final collage remains sharp.
PSS_GEOMETRY = [
    (1, 42, 42, 42, 34), (2, 84, 42, 42, 34),
    (3, 38, 76, 46, 34), (4, 88, 76, 42, 34),
    (5, 32, 110, 49, 37), (6, 81, 110, 46, 37),
    (7, 127, 110, 43, 37), (8, 170, 110, 48, 37),
    (9, 28, 147, 53, 40), (10, 81, 147, 48, 40),
    (11, 129, 147, 46, 40), (12, 175, 147, 48, 40),
    (13, 25, 187, 53, 40), (14, 78, 187, 51, 40),
    (15, 129, 187, 48, 40), (16, 177, 187, 47, 40),
    (17, 24, 227, 54, 40), (18, 78, 227, 51, 40),
    (19, 129, 227, 48, 40), (20, 177, 227, 47, 40),
    (21, 26, 267, 53, 39), (22, 79, 267, 50, 39),
    (23, 129, 267, 48, 39), (24, 177, 267, 47, 39),
    (25, 31, 306, 51, 40), (26, 82, 306, 47, 40),
    (27, 129, 306, 48, 40), (28, 177, 306, 47, 40),
    (29, 53, 346, 43, 37), (30, 96, 346, 43, 37),
    (31, 145, 346, 50, 37),
]

# -----------------------------
# Session state
# -----------------------------
if "sheets" not in st.session_state:
    st.session_state.sheets = [
        {
            "title": "",
            "month": "Aug-26",
            "mal_capacity": "",
            "no_pss": "",
            "total_capacity": "",
            "colors": {i: None for i in range(1, N_PSS + 1)},
            "dates": {i: None for i in range(1, N_PSS + 1)},
        }
        for _ in range(N_SHEETS)
    ]

if "active_sheet" not in st.session_state:
    st.session_state.active_sheet = 0

if "selected_date" not in st.session_state:
    st.session_state.selected_date = date.today()

if "selected_color" not in st.session_state:
    st.session_state.selected_color = "Yellow"

if "selected_sheet_for_controls" not in st.session_state:
    st.session_state.selected_sheet_for_controls = 0


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value):
    return html.escape(str(value or ""))


def parse_date_from_string(value):
    if not value:
        return None
    return str(value)


def make_svg(sheet_idx, editable=True):
    sheet = st.session_state.sheets[sheet_idx]
    title = sheet["title"] or f"Solar - Sheet {sheet_idx + 1}"
    month = sheet["month"] or "Aug-26"

    # Sheet dimensions in SVG units.
    W, H = 245, 500

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" style="display:block;width:100%;height:auto;">',
        '<rect x="2" y="2" width="241" height="496" rx="1" fill="white" stroke="#555" stroke-width="1.2"/>',

        # Header
        f'<text x="12" y="24" font-family="Arial, sans-serif" font-size="12" '
        f'font-weight="600" fill="#111">{clean_text(title)}</text>',
        f'<text x="197" y="24" text-anchor="end" font-family="Arial, sans-serif" '
        f'font-size="11" fill="#111">{clean_text(month)}</text>',
        '<line x1="10" y1="29" x2="235" y2="29" stroke="#999" stroke-width="0.7"/>',

        # Main plant silhouette
        '<path d="M30 45 Q53 31 84 34 L163 34 Q193 36 213 55 '
        'L211 350 Q205 397 174 420 L145 438 L105 438 L72 421 '
        'Q39 401 28 353 Z" fill="#fff" stroke="#111" stroke-width="2.2"/>',

        # Central divider
        '<path d="M121 38 Q118 100 120 160 Q121 225 122 290 Q123 355 129 430" '
        'fill="none" stroke="#111" stroke-width="3.2"/>',
    ]

    # PSS blocks
    for pss, x, y, w, h in PSS_GEOMETRY:
        fill = sheet["colors"].get(pss) or "#FFFFFF"
        date_label = sheet["dates"].get(pss)
        cx = x + w / 2
        cy = y + h / 2

        # Slightly irregular rounded polygon to feel like the paper sketch.
        skew = 2 if pss % 3 == 0 else -1
        points = (
            f"{x+skew},{y+2} {x+w-2},{y} {x+w},{y+h-3} "
            f"{x+w-3},{y+h} {x+2},{y+h-1}"
        )

        if editable:
            # Clickable SVG link. It sends the PSS number through query parameters.
            href = f"?sheet={sheet_idx}&pss={pss}&_click=1"
            svg.append(
                f'<a href="{href}" title="PSS {pss}">'
                f'<polygon points="{points}" fill="{fill}" stroke="#111" stroke-width="1.1"/>'
                f'<text x="{cx}" y="{cy+4}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" font-weight="600" fill="#111">'
                f'{pss}</text></a>'
            )
        else:
            svg.extend([
                f'<polygon points="{points}" fill="{fill}" stroke="#111" stroke-width="1.1"/>',
                f'<text x="{cx}" y="{cy+4}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" font-weight="600" fill="#111">'
                f'{pss}</text>',
            ])

    # Footer
    svg.extend([
        '<line x1="10" y1="445" x2="235" y2="445" stroke="#999" stroke-width="0.7"/>',
        '<text x="12" y="462" font-family="Arial, sans-serif" font-size="8.5" fill="#222">'
        'MAL Capacity (MW)</text>',
        '<line x1="75" y1="460" x2="142" y2="460" stroke="#333" stroke-width="0.7"/>',
        '<text x="150" y="462" font-family="Arial, sans-serif" font-size="8.5" fill="#222">'
        'No. of PSS</text>',
        '<line x1="195" y1="460" x2="232" y2="460" stroke="#333" stroke-width="0.7"/>',

        '<text x="12" y="481" font-family="Arial, sans-serif" font-size="8.5" fill="#222">'
        'Total Capacity (MW)</text>',
        '<line x1="78" y1="479" x2="150" y2="479" stroke="#333" stroke-width="0.7"/>',
    ])

    # Filled footer values, if supplied.
    if sheet["mal_capacity"]:
        svg.append(
            f'<text x="108" y="462" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="8.5" font-weight="600">{clean_text(sheet["mal_capacity"])}</text>'
        )
    if sheet["no_pss"]:
        svg.append(
            f'<text x="214" y="462" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="8.5" font-weight="600">{clean_text(sheet["no_pss"])}</text>'
        )
    if sheet["total_capacity"]:
        svg.append(
            f'<text x="114" y="481" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="8.5" font-weight="600">{clean_text(sheet["total_capacity"])}</text>'
        )

    svg.append("</svg>")
    return "".join(svg)


def make_collage_svg():
    # 5 columns x 2 rows, similar to the physical collage.
    cols, rows = 5, 2
    sheet_w, sheet_h = 245, 500
    gap_x, gap_y = 16, 20
    margin = 18

    W = margin * 2 + cols * sheet_w + (cols - 1) * gap_x
    H = margin * 2 + rows * sheet_h + (rows - 1) * gap_y

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" style="display:block;width:100%;height:auto;background:#eee;">'
    ]

    for i in range(N_SHEETS):
        col = i % cols
        row = i // cols
        x = margin + col * (sheet_w + gap_x)
        y = margin + row * (sheet_h + gap_y)

        sheet_svg = make_svg(i, editable=False)
        # Remove outer svg wrapper so it can be nested.
        inner = re.sub(r"^<svg[^>]*>", "", sheet_svg)
        inner = re.sub(r"</svg>$", "", inner)

        out.append(f'<g transform="translate({x},{y})">{inner}</g>')

    out.append("</svg>")
    return "".join(out)


def svg_to_downloadable_data(svg):
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# -----------------------------
# Apply click from SVG
# -----------------------------
params = st.query_params
if "sheet" in params and "pss" in params and "_click" in params:
    try:
        clicked_sheet = int(params["sheet"])
        clicked_pss = int(params["pss"])
        if 0 <= clicked_sheet < N_SHEETS and 1 <= clicked_pss <= N_PSS:
            target = st.session_state.sheets[clicked_sheet]
            target["colors"][clicked_pss] = PALETTE[st.session_state.selected_color]
            target["dates"][clicked_pss] = parse_date_from_string(
                st.session_state.selected_date
            )
            st.session_state.active_sheet = clicked_sheet
            st.session_state.selected_sheet_for_controls = clicked_sheet
    except Exception:
        pass

    # Clear click parameters after processing so refresh doesn't re-apply.
    try:
        st.query_params.clear()
    except Exception:
        pass


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .title {
        font-size: 28px;
        font-weight: 750;
        margin-bottom: 2px;
    }
    .subtitle {
        color: #666;
        margin-bottom: 18px;
    }
    .control-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 14px 16px;
        background: #fafafa;
    }
    .sheet-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 8px;
        background: white;
    }
    .small-note {
        color: #666;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">☀️ Solar PSS Coloring & Sheet Collage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Select a date and color, then click the PSS blocks you want to fill. '
    'Complete up to 10 sheets and generate the final collage.</div>',
    unsafe_allow_html=True,
)

# Main controls
c1, c2, c3, c4 = st.columns([1.25, 1.1, 1.2, 1.2])

with c1:
    st.session_state.selected_sheet_for_controls = st.selectbox(
        "Sheet",
        list(range(N_SHEETS)),
        index=st.session_state.selected_sheet_for_controls,
        format_func=lambda x: f"Sheet {x + 1}",
    )
    st.session_state.active_sheet = st.session_state.selected_sheet_for_controls

with c2:
    st.session_state.selected_date = st.date_input(
        "Date",
        value=st.session_state.selected_date,
        format="DD-MMM-YYYY",
    )

with c3:
    st.session_state.selected_color = st.selectbox(
        "Color",
        list(PALETTE.keys()),
        index=list(PALETTE.keys()).index(st.session_state.selected_color),
    )

with c4:
    st.markdown("**Current color**")
    st.markdown(
        f'<div style="height:38px;border-radius:8px;border:1px solid #999;'
        f'background:{PALETTE[st.session_state.selected_color]};"></div>',
        unsafe_allow_html=True,
    )

# Sheet information
idx = st.session_state.active_sheet
sheet = st.session_state.sheets[idx]

st.markdown("### Sheet information")
i1, i2, i3, i4, i5 = st.columns([1.6, 1.0, 1.0, 1.0, 1.0])

with i1:
    sheet["title"] = st.text_input(
        "Left header",
        value=sheet["title"],
        placeholder="e.g. Solar - G1",
        key=f"title_{idx}",
    )

with i2:
    sheet["month"] = st.text_input(
        "Right header",
        value=sheet["month"],
        placeholder="e.g. Aug-26",
        key=f"month_{idx}",
    )

with i3:
    sheet["mal_capacity"] = st.text_input(
        "MAL Capacity (MW)",
        value=sheet["mal_capacity"],
        key=f"mal_{idx}",
    )

with i4:
    sheet["no_pss"] = st.text_input(
        "No. of PSS",
        value=sheet["no_pss"],
        key=f"npss_{idx}",
    )

with i5:
    sheet["total_capacity"] = st.text_input(
        "Total Capacity (MW)",
        value=sheet["total_capacity"],
        key=f"total_{idx}",
    )

st.markdown(
    f'**Sheet {idx + 1}:** Select `{st.session_state.selected_date.strftime("%d-%b-%Y")}` '
    f'and `{st.session_state.selected_color}`, then click PSS blocks below.',
    unsafe_allow_html=True,
)

# Display editable sheet
left, right = st.columns([1.1, 1.0])

with left:
    st.markdown("### Interactive sheet")
    st.components.v1.html(
        make_svg(idx, editable=True),
        height=525,
        scrolling=False,
    )

with right:
    st.markdown("### Current date assignments")
    assignments = []
    for pss in range(1, N_PSS + 1):
        d = sheet["dates"].get(pss)
        c = sheet["colors"].get(pss)
        if d:
            assignments.append(
                {
                    "PSS": pss,
                    "Date": d,
                    "Color": next(
                        (name for name, hexv in PALETTE.items() if hexv == c),
                        "Custom",
                    ),
                }
            )

    if assignments:
        st.dataframe(assignments, use_container_width=True, hide_index=True)
    else:
        st.info("No PSS has been colored on this sheet yet.")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Clear this sheet", use_container_width=True):
            sheet["colors"] = {i: None for i in range(1, N_PSS + 1)}
            sheet["dates"] = {i: None for i in range(1, N_PSS + 1)}
            st.rerun()

    with b2:
        if st.button("Clear all sheets", use_container_width=True):
            for s in st.session_state.sheets:
                s["colors"] = {i: None for i in range(1, N_PSS + 1)}
                s["dates"] = {i: None for i in range(1, N_PSS + 1)}
            st.rerun()

# Sheet navigation
st.markdown("### Sheets")
sheet_cols = st.columns(10)
for i in range(N_SHEETS):
    with sheet_cols[i]:
        if st.button(
            f"{i + 1}",
            key=f"sheet_nav_{i}",
            type="primary" if i == st.session_state.active_sheet else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_sheet = i
            st.session_state.selected_sheet_for_controls = i
            st.rerun()

# Preview all 10 sheets
st.markdown("## Final 10-sheet collage")
st.caption("The collage below is generated from the 10 sheets. It is arranged 5 × 2 by default.")

st.components.v1.html(
    make_collage_svg(),
    height=1080,
    scrolling=True,
)

# Download SVG files
st.markdown("### Export")
collage_svg = make_collage_svg()
st.download_button(
    "⬇️ Download 10-sheet collage (SVG)",
    data=collage_svg,
    file_name="solar_10_sheet_collage.svg",
    mime="image/svg+xml",
    use_container_width=True,
)

# Legend
st.markdown("### Color legend")
legend_cols = st.columns(len(PALETTE))
for col, (name, color) in zip(legend_cols, PALETTE.items()):
    with col:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:7px;">'
            f'<span style="width:18px;height:18px;background:{color};'
            f'border:1px solid #777;border-radius:3px;display:inline-block;"></span>'
            f'<span>{html.escape(name)}</span></div>',
            unsafe_allow_html=True,
        )
