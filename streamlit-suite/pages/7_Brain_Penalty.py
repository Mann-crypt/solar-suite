import io
from datetime import date

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Solar PSS Sheet Painter", page_icon="☀️", layout="wide")

# ============================================================
# CONFIG
# ============================================================
N_SHEETS = 10
N_PSS = 31
PALETTE = {
    "Red": "#F26B4F",
    "Yellow": "#FFF36A",
    "Green": "#35C98B",
    "Blue": "#2F80ED",
    "Orange": "#F5A623",
    "Purple": "#A970FF",
    "Teal": "#28C7C7",
    "White": "#FFFFFF",
}

# PSS rows follow the photographed sheet: 1-2, 3-4, then 4 columns,
# ending with 29-31. Coordinates are deliberately fixed so every sheet
# has exactly the same geometry.
ROWS = [, [3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
, [17, 18, 19, 20], [21, 22, 23, 24],
, [29, 30, 31],
]
Y = [145, 235, 325, 415, 505, 595, 685, 775, 865, 955]
FOUR_X = [(145, 300), (300, 455), (545, 700), (700, 855)]


def make_polygons():
    polys = {}
    for ri, row in enumerate(ROWS):
        top, bottom = Y[ri], Y[ri + 1]
        if len(row) == 2:
            xs = [(320, 470), (530, 680)]
        elif len(row) == 3:
            xs = [(245, 395), (395, 545), (545, 695)]
        else:
            xs = FOUR_X
        for i, pss in enumerate(row):
            x1, x2 = xs[i]
            slant = 8 if ri < 2 else 4
            polys[pss] = [
                (x1 + slant, top), (x2 - slant, top + 2),
                (x2, bottom - 3), (x1, bottom),
            ]
    return polys


POLYS = make_polygons()


def center(points):
    return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))


def empty_sheet():
    return {
        "name": "",
        "header_date": date.today(),
        "mal": "",
        "total": "",
        "assignments": {},  # pss -> {date: ISO string, color: hex}
    }


if "sheets" not in st.session_state:
    st.session_state.sheets = [empty_sheet() for _ in range(N_SHEETS)]
if "active_sheet" not in st.session_state:
    st.session_state.active_sheet = 0
if "paint_date" not in st.session_state:
    st.session_state.paint_date = date.today()
if "paint_color_name" not in st.session_state:
    st.session_state.paint_color_name = "Red"
if "paint_custom" not in st.session_state:
    st.session_state.paint_custom = PALETTE["Red"]
if "selected_pss" not in st.session_state:
    st.session_state.selected_pss = set()
if "processed_event" not in st.session_state:
    st.session_state.processed_event = None
if "last_synced_paint_date" not in st.session_state:
    st.session_state.last_synced_paint_date = st.session_state.paint_date.isoformat()


def current_sheet():
    return st.session_state.sheets[st.session_state.active_sheet]


def assignments_for_date(sheet, d):
    iso = d.isoformat()
    return {p for p, v in sheet["assignments"].items() if v["date"] == iso}


def make_map(sheet, selected_date):
    fig = go.Figure()
    d_iso = selected_date.isoformat()
    for pss in range(1, N_PSS + 1):
        pts = POLYS[pss]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        assignment = sheet["assignments"].get(pss)
        fill = assignment["color"] if assignment else PALETTE["Red"]
        opacity = 0.97 if assignment else 0.88
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=fill, line=dict(color="black", width=2),
            customdata=[pss] * len(xs), hovertemplate=f"PSS {pss}<extra></extra>",
            name=f"PSS {pss}", showlegend=False, opacity=opacity,
        ))
        cx, cy = center(pts)
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers+text",
            text=[str(pss)], textposition="middle center",
            marker=dict(size=34, color="rgba(255,255,255,0.01)", line=dict(width=0)),
            textfont=dict(size=15, color="black"),
            customdata=[pss], hovertemplate=f"Click PSS {pss}<extra></extra>",
            name=f"PSS {pss} click", showlegend=False,
        ))

    fig.update_xaxes(visible=False, range=[60, 940], fixedrange=True)
    fig.update_yaxes(visible=False, range=[1120, 70], fixedrange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        clickmode="event+select",
        dragmode=False,
    )
    return fig


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_sheet(sheet, width=900):
    W, H = 1000, 1300
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = load_font(35, True)
    small = load_font(20)
    pfont = load_font(23, True)

    title = sheet["name"] or "Solar - ____"
    month = sheet["header_date"].strftime("%b-%y")
    d.text((60, 48), title, fill="black", font=title_font)
    d.line((55, 95, 250, 95), fill="black", width=2)
    bb = d.textbbox((0, 0), month, font=title_font)
    d.text((W - 60 - (bb[2] - bb[0]), 48), month, fill="black", font=title_font)

    outer = [
        (130, 150), (225, 100), (450, 92), (500, 115), (550, 92),
        (775, 100), (870, 150), (905, 350), (920, 600), (900, 850),
        (825, 1030), (690, 1110), (560, 1125), (500, 1090), (440, 1125),
        (310, 1110), (175, 1030), (100, 850), (80, 600), (95, 350),
    ]
    d.polygon(outer, fill="white", outline="black")

    for pss, pts in POLYS.items():
        v = sheet["assignments"].get(pss)
        fill = v["color"] if v else PALETTE["Red"]
        d.polygon(pts, fill=fill, outline="black")
        cx, cy = center(pts)
        txt = str(pss)
        bb = d.textbbox((0, 0), txt, font=pfont)
        d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2), txt, fill="black", font=pfont)

    d.line(outer + [outer[0]], fill="black", width=6, joint="curve")
    d.line((500, 112, 500, 1095), fill="black", width=8)

    fy = 1160
    mal = sheet["mal"] or "________"
    total = sheet["total"] or "________"
    assigned_count = len(sheet["assignments"])
    d.text((60, fy), f"MAL Capacity (MW): {mal}", fill="black", font=small)
    d.text((600, fy), f"No. of PSS: {assigned_count}", fill="black", font=small)
    d.text((60, fy + 48), f"Total Capacity (MW): {total}", fill="black", font=small)
    return img


def collage(images, columns=4):
    gap = 24
    thumb_w = 520
    thumb_h = 676
    rows = int(np.ceil(len(images) / columns))
    out = Image.new("RGB", (columns * thumb_w + (columns + 1) * gap,
                             rows * thumb_h + (rows + 1) * gap), "#E9E9E9")
    for i, im in enumerate(images):
        cp = im.copy()
        cp.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = gap + (i % columns) * (thumb_w + gap) + (thumb_w - cp.width) // 2
        y = gap + (i // columns) * (thumb_h + gap) + (thumb_h - cp.height) // 2
        out.paste(cp, (x, y))
    return out


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


# ============================================================
# UI
# ============================================================
st.title("☀️ Solar PSS Sheet Painter")
st.caption("Select a date and color, then click PSS blocks on the map. Create 10 sheets and export a 4-4-2 collage.")

# Sheet Navigation Header Tabs
sheet_cols = st.columns(N_SHEETS)
for i, col in enumerate(sheet_cols):
    with col:
        sheet_item = st.session_state.sheets[i]
        # Resolve the KeyError securely here:
        sheet_name = sheet_item.get("name")
        label = f"📄 {sheet_name}" if sheet_name else f"Sheet {i + 1}"
        
        # Highlight active sheet button style
        if st.session_state.active_sheet == i:
            st.markdown(f"**🎯 {label}**")
        else:
            if st.button(label, key=f"tab_btn_{i}"):
                st.session_state.active_sheet = i
                st.rerun()

st.divider()

# Split work arena: Left sidebar parameters / Right map visualizer
sheet = current_sheet()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Sheet Details")
    sheet["name"] = st.text_input("Sheet Name / Project Title", value=sheet["name"], key=f"s_name_{st.session_state.active_sheet}")
    sheet["header_date"] = st.date_input("Target Month/Date", value=sheet["header_date"], key=f"s_date_{st.session_state.active_sheet}")
    sheet["mal"] = st.text_input("MAL Capacity (MW)", value=sheet["mal"], key=f"s_mal_{st.session_state.active_sheet}")
    sheet["total"] = st.text_input("Total Capacity (MW)", value=sheet["total"], key=f"s_tot_{st.session_state.active_sheet}")
    
    st.divider()
    st.subheader("🎨 Paint Settings")
    p_date = st.date_input("Assignment Paint Date", value=st.session_state.paint_date)
    st.session_state.paint_date = p_date
    
    color_choice = st.selectbox("Paint Brush Color", list(PALETTE.keys()), index=list(PALETTE.keys()).index(st.session_state.paint_color_name))
    st.session_state.paint_color_name = color_choice
    chosen_hex = PALETTE[color_choice]

with col2:
    st.subheader(f"🗺️ Map Interface: {sheet['name'] or f'Sheet {st.session_state.active_sheet + 1}'}")
    fig = make_map(sheet, st.session_state.paint_date)
    
    # Process clicks natively through Plotly Events
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    # Check selection point payload dynamically
