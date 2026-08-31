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
ROWS = [
    [1, 2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
    [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24],
    [25, 26, 27, 28], [29, 30, 31],
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


def normalize_sheet(raw):
    """Repair old/incomplete session-state sheet objects safely."""
    base = empty_sheet()
    if not isinstance(raw, dict):
        return base
    for key in base:
        if key in raw:
            base[key] = raw[key]
    if not isinstance(base["assignments"], dict):
        base["assignments"] = {}
    clean = {}
    for pss, value in base["assignments"].items():
        try:
            pss_int = int(pss)
        except (TypeError, ValueError):
            continue
        if not 1 <= pss_int <= N_PSS or not isinstance(value, dict):
            continue
        if "date" not in value or "color" not in value:
            continue
        clean[pss_int] = {"date": str(value["date"]), "color": str(value["color"])}
    base["assignments"] = clean
    return base


# Streamlit can preserve session_state across code edits/redeploys.
# Normalize every sheet so stale/incomplete dictionaries never cause KeyError.
if "sheets" not in st.session_state or not isinstance(st.session_state.sheets, list):
    st.session_state.sheets = []

st.session_state.sheets = [
    normalize_sheet(s) for s in st.session_state.sheets[:N_SHEETS]
]
while len(st.session_state.sheets) < N_SHEETS:
    st.session_state.sheets.append(empty_sheet())

if "active_sheet" not in st.session_state:
    st.session_state.active_sheet = 0
st.session_state.active_sheet = max(
    0, min(int(st.session_state.active_sheet), N_SHEETS - 1)
)
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
if "map_version" not in st.session_state:
    st.session_state.map_version = 0
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
        # Only show assigned colors. Unassigned defaults to the photographed orange/red.
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

    # Draw all PSS cells. Unassigned cells retain the red/orange paper-map look.
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

# Sheet tabs
sheet_cols = st.columns(N_SHEETS)
for i, col in enumerate(sheet_cols):
    with col:
        label = st.session_state.sheets[i]["name"] or f"Sheet {i + 1}"
        if st.button(label, key=f"sheet_{i}", width="stretch"):
            st.session_state.active_sheet = i
            st.session_state.selected_pss = assignments_for_date(st.session_state.sheets[i], st.session_state.paint_date)
            st.session_state.last_synced_paint_date = st.session_state.paint_date.isoformat()
            st.session_state.processed_event = None
            st.session_state.map_version += 1
            st.rerun()

st.divider()
idx = st.session_state.active_sheet
sheet = current_sheet()

controls, preview = st.columns([0.9, 1.7], gap="large")
with controls:
    st.subheader(f"Sheet {idx + 1}")
    sheet["name"] = st.text_input("Write on left header", value=sheet["name"], key=f"name_{idx}", placeholder="Solar - G1")
    sheet["header_date"] = st.date_input("Header month/date", value=sheet["header_date"], key=f"header_date_{idx}")
    sheet["mal"] = st.text_input("MAL Capacity (MW)", value=sheet["mal"], key=f"mal_{idx}")
    sheet["total"] = st.text_input("Total Capacity (MW)", value=sheet["total"], key=f"total_{idx}")

    st.markdown("### Paint by date")
    st.session_state.paint_date = st.date_input("1. Click/select date", value=st.session_state.paint_date, key="global_paint_date")
    if st.session_state.paint_date.isoformat() != st.session_state.last_synced_paint_date:
        st.session_state.selected_pss = assignments_for_date(sheet, st.session_state.paint_date)
        st.session_state.last_synced_paint_date = st.session_state.paint_date.isoformat()
        st.session_state.processed_event = None
    st.session_state.paint_color_name = st.selectbox("2. Select color", list(PALETTE), index=list(PALETTE).index(st.session_state.paint_color_name), key="global_color")
    st.session_state.paint_custom = st.color_picker("Optional: custom fill color", value=PALETTE[st.session_state.paint_color_name], key="global_custom")

    st.write(f"**Selected for {st.session_state.paint_date.strftime('%d-%b-%Y')}:** {', '.join(map(str, sorted(st.session_state.selected_pss))) or 'None'}")

    if st.button("Apply selected PSS", type="primary", width="stretch"):
        color = st.session_state.paint_custom
        d_iso = st.session_state.paint_date.isoformat()
        # Preserve other dates on every PSS. For this selected date, assign/remove exactly the selected set.
        for pss in range(1, N_PSS + 1):
            if pss in st.session_state.selected_pss:
                sheet["assignments"][pss] = {"date": d_iso, "color": color}
            elif sheet["assignments"].get(pss, {}).get("date") == d_iso:
                del sheet["assignments"][pss]
        st.success("Color assignment saved.")
        st.session_state.processed_event = None
        st.session_state.map_version += 1
        st.rerun()

    if st.button("Clear selected date", width="stretch"):
        d_iso = st.session_state.paint_date.isoformat()
        sheet["assignments"] = {p: v for p, v in sheet["assignments"].items() if v["date"] != d_iso}
        st.session_state.selected_pss = set()
        st.session_state.processed_event = None
        st.session_state.map_version += 1
        st.rerun()

    if st.button("Clear this sheet", width="stretch"):
        sheet["assignments"] = {}
        st.session_state.selected_pss = set()
        st.session_state.processed_event = None
        st.session_state.map_version += 1
        st.rerun()

    st.markdown("### Assigned dates")
    if sheet["assignments"]:
        rows = []
        for pss, v in sorted(sheet["assignments"].items()):
            rows.append({"PSS": pss, "Date": v["date"], "Color": v["color"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No PSS dates assigned yet.")

with preview:
    st.subheader("Click PSS blocks to paint")
    fig = make_map(sheet, st.session_state.paint_date)
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"pss_map_{idx}_{st.session_state.map_version}",
        on_select="rerun",
        selection_mode=["points"],
    )

    # Plotly selection is used as the click-to-select mechanism. Center markers make
    # every PSS reliably clickable while the filled polygons preserve the visual map.
    if event is not None:
        points = getattr(getattr(event, "selection", None), "points", [])
        if points:
            pss_values = []
            for point in points:
                cd = point.get("customdata") if isinstance(point, dict) else None
                if cd is not None:
                    try:
                        pss_values.append(int(cd[0] if isinstance(cd, (list, tuple)) else cd))
                    except (TypeError, ValueError):
                        pass
            if pss_values:
                # Process only a new event so Streamlit reruns don't repeatedly toggle the same click.
                sig = (idx, st.session_state.paint_date.isoformat(), tuple(sorted(pss_values)))
                if sig != st.session_state.processed_event:
                    selected = set(st.session_state.selected_pss)
                    for pss in pss_values:
                        if pss in selected:
                            selected.remove(pss)
                        else:
                            selected.add(pss)
                    st.session_state.selected_pss = selected
                    st.session_state.processed_event = sig
                    # Changing the chart key clears the previous Plotly selection.
                    st.session_state.map_version += 1
                    st.rerun()

    st.caption("Click a PSS number. Click it again to remove it from the current date selection. Then press 'Apply selected PSS'.")

# ============================================================
# FINAL COLLAGE
# ============================================================
st.divider()
st.subheader("Final 10-sheet collage")
st.caption("The collage is arranged 4 + 4 + 2, matching the layout of your photograph.")

if st.button("Generate collage", type="primary", width="stretch"):
    st.session_state.make_collage = True

if st.session_state.get("make_collage", False):
    images = [render_sheet(s) for s in st.session_state.sheets]
    final = collage(images, columns=4)
    st.image(final, width="stretch")
    st.download_button(
        "⬇️ Download collage PNG",
        data=png_bytes(final),
        file_name="solar_pss_10_sheet_collage.png",
        mime="image/png",
        width="stretch",
    )

# Export mapping for backup / Excel workflow.
rows = []
for i, s in enumerate(st.session_state.sheets, start=1):
    for pss in range(1, N_PSS + 1):
        v = s["assignments"].get(pss)
        rows.append({
            "Sheet": i,
            "Header": s["name"],
            "Header Date": s["header_date"].isoformat(),
            "PSS": pss,
            "Assigned Date": v["date"] if v else "",
            "Color": v["color"] if v else "",
            "MAL Capacity (MW)": s["mal"],
            "Total Capacity (MW)": s["total"],
        })
backup = pd.DataFrame(rows)
st.download_button("Download PSS/date mapping CSV", backup.to_csv(index=False), "solar_pss_mapping.csv", "text/csv")
