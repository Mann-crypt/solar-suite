import io
from datetime import date

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Solar Date Coloring", page_icon="☀️", layout="wide")

# ============================================================
# CONFIG
# ============================================================
N_SHEETS = 10
N_DAYS = 31
COLORS = {
    "Red": "#F26B4F",
    "Green": "#35C98B",
    "Yellow": "#FFF36A",
    "Blue": "#2F80ED",
}
DEFAULT_FILL = "#F26B4F"

# These are DATE BLOCKS, not PSS numbers.
ROWS = [
    [1, 2],
    [3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31],
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
        for i, day in enumerate(row):
            x1, x2 = xs[i]
            slant = 8 if ri < 2 else 4
            polys[day] = [
                (x1 + slant, top), (x2 - slant, top + 2),
                (x2, bottom - 3), (x1, bottom),
            ]
    return polys


POLYS = make_polygons()


def center(points):
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


def empty_sheet():
    return {
        "name": "",
        "header_date": date.today().replace(day=1),
        "mal": "",
        "total": "",
        # day -> {color_name, color_hex}
        "days": {},
    }


def normalize_sheet(raw):
    """Repair old session state and convert old PSS-style data safely."""
    base = empty_sheet()
    if not isinstance(raw, dict):
        return base

    base["name"] = str(raw.get("name", "") or "")
    raw_date = raw.get("header_date", base["header_date"])
    if isinstance(raw_date, date):
        base["header_date"] = raw_date.replace(day=1)
    else:
        try:
            base["header_date"] = date.fromisoformat(str(raw_date))
            base["header_date"] = base["header_date"].replace(day=1)
        except Exception:
            base["header_date"] = date.today().replace(day=1)

    base["mal"] = str(raw.get("mal", "") or "")
    base["total"] = str(raw.get("total", "") or "")

    source = raw.get("days")
    if not isinstance(source, dict):
        # If a previous version stored assignments as PSS -> {date,color},
        # don't carry those old semantics into the new date model.
        source = {}

    clean = {}
    for day, value in source.items():
        try:
            day = int(day)
        except (TypeError, ValueError):
            continue
        if not 1 <= day <= N_DAYS or not isinstance(value, dict):
            continue
        color_name = str(value.get("color_name", "") or "")
        color_hex = str(value.get("color_hex", "") or "")
        if color_name not in COLORS:
            # Accept old hex only when it exactly matches one of our four colors.
            matches = [k for k, v in COLORS.items() if v.lower() == color_hex.lower()]
            if not matches:
                continue
            color_name = matches[0]
        clean[day] = {
            "color_name": color_name,
            "color_hex": COLORS[color_name],
        }
    base["days"] = clean
    return base


# ============================================================
# SESSION STATE
# ============================================================
if "sheets" not in st.session_state or not isinstance(st.session_state.sheets, list):
    st.session_state.sheets = []

st.session_state.sheets = [
    normalize_sheet(s) for s in st.session_state.sheets[:N_SHEETS]
]
while len(st.session_state.sheets) < N_SHEETS:
    st.session_state.sheets.append(empty_sheet())

if "active_sheet" not in st.session_state:
    st.session_state.active_sheet = 0
st.session_state.active_sheet = max(0, min(int(st.session_state.active_sheet), N_SHEETS - 1))

if "selected_day" not in st.session_state:
    st.session_state.selected_day = min(date.today().day, N_DAYS)
if "selected_color" not in st.session_state:
    st.session_state.selected_color = "Red"
if "map_version" not in st.session_state:
    st.session_state.map_version = 0
if "last_click_signature" not in st.session_state:
    st.session_state.last_click_signature = None
if "show_collage" not in st.session_state:
    st.session_state.show_collage = False


def current_sheet():
    return st.session_state.sheets[st.session_state.active_sheet]


def actual_date(sheet, day):
    """Convert day number to the actual date in that sheet's header month."""
    y = sheet["header_date"].year
    m = sheet["header_date"].month
    try:
        return date(y, m, day)
    except ValueError:
        return None


def make_map(sheet):
    fig = go.Figure()
    for day in range(1, N_DAYS + 1):
        pts = POLYS[day]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        assigned = sheet["days"].get(day)
        fill = assigned["color_hex"] if assigned else DEFAULT_FILL

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(color="black", width=2),
                customdata=[day] * len(xs),
                hovertemplate=f"Date {day}<extra></extra>",
                showlegend=False,
                opacity=0.97,
            )
        )
        cx, cy = center(pts)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                text=[str(day)],
                textposition="middle center",
                marker=dict(size=40, color="rgba(255,255,255,0.01)", line=dict(width=0)),
                textfont=dict(size=15, color="black"),
                customdata=[day],
                hovertemplate=f"Click date {day}<extra></extra>",
                showlegend=False,
            )
        )

    # Plant outline / center split, matching the paper-style page.
    outer = [
        (130, 150), (225, 100), (450, 92), (500, 115), (550, 92),
        (775, 100), (870, 150), (905, 350), (920, 600), (900, 850),
        (825, 1030), (690, 1110), (560, 1125), (500, 1090), (440, 1125),
        (310, 1110), (175, 1030), (100, 850), (80, 600), (95, 350),
    ]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in outer] + [outer[0][0]],
        y=[p[1] for p in outer] + [outer[0][1]],
        mode="lines", line=dict(color="black", width=5),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[500, 500], y=[112, 1095], mode="lines",
        line=dict(color="black", width=7), hoverinfo="skip", showlegend=False,
    ))

    fig.update_xaxes(visible=False, range=[60, 940], fixedrange=True)
    fig.update_yaxes(visible=False, range=[1120, 70], fixedrange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        clickmode="event+select",
        dragmode=False,
        hovermode="closest",
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


def render_sheet(sheet):
    W, H = 1000, 1300
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = load_font(35, True)
    small = load_font(20)
    day_font = load_font(23, True)

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

    for day, pts in POLYS.items():
        v = sheet["days"].get(day)
        fill = v["color_hex"] if v else DEFAULT_FILL
        d.polygon(pts, fill=fill, outline="black")
        cx, cy = center(pts)
        txt = str(day)
        bb = d.textbbox((0, 0), txt, font=day_font)
        d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2), txt, fill="black", font=day_font)

    d.line(outer + [outer[0]], fill="black", width=6, joint="curve")
    d.line((500, 112, 500, 1095), fill="black", width=8)

    fy = 1160
    mal = sheet["mal"] or "________"
    total = sheet["total"] or "________"
    d.text((60, fy), f"MAL Capacity (MW): {mal}", fill="black", font=small)
    d.text((600, fy), f"No. of PSS: ________", fill="black", font=small)
    d.text((60, fy + 48), f"Total Capacity (MW): {total}", fill="black", font=small)
    return img


def collage(images, columns=4):
    gap = 24
    thumb_w = 520
    thumb_h = 676
    rows = int(np.ceil(len(images) / columns))
    out = Image.new(
        "RGB",
        (columns * thumb_w + (columns + 1) * gap, rows * thumb_h + (rows + 1) * gap),
        "#E9E9E9",
    )
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
st.title("☀️ Solar Date Coloring")
st.caption("The numbered blocks are dates (1–31), not PSS. Choose a date block, choose one of four colors, and save it. Colors remain saved for the next day.")

# Sheet navigation
sheet_cols = st.columns(N_SHEETS)
for i, col in enumerate(sheet_cols):
    with col:
        label = st.session_state.sheets[i]["name"] or f"Sheet {i + 1}"
        if st.button(label, key=f"sheet_{i}", width="stretch"):
            st.session_state.active_sheet = i
            st.session_state.last_click_signature = None
            st.session_state.map_version += 1
            st.rerun()

st.divider()
idx = st.session_state.active_sheet
sheet = current_sheet()

controls, preview = st.columns([0.75, 1.65], gap="large")

with controls:
    st.subheader(f"Sheet {idx + 1}")

    sheet["name"] = st.text_input(
        "Write on left header",
        value=sheet["name"],
        key=f"name_{idx}",
        placeholder="Solar - G1",
    )
    sheet["header_date"] = st.date_input(
        "Header month",
        value=sheet["header_date"],
        key=f"header_date_{idx}",
    ).replace(day=1)
    sheet["mal"] = st.text_input("MAL Capacity (MW)", value=sheet["mal"], key=f"mal_{idx}")
    sheet["total"] = st.text_input("Total Capacity (MW)", value=sheet["total"], key=f"total_{idx}")

    st.markdown("### Color dates")
    st.info("Click a numbered date on the map. Then choose Red, Green, Yellow, or Blue.")

    selected_day = st.session_state.selected_day
    selected_day_date = actual_date(sheet, selected_day)
    if selected_day_date:
        st.write(f"**Selected date: {selected_day_date.strftime('%d-%b-%Y')} (day {selected_day})**")
    else:
        st.write(f"**Selected day: {selected_day}** (outside this month)")

    color_cols = st.columns(4)
    for color_name, col in zip(COLORS, color_cols):
        with col:
            if st.button(
                color_name,
                key=f"color_{color_name}",
                width="stretch",
                type="primary" if st.session_state.selected_color == color_name else "secondary",
            ):
                st.session_state.selected_color = color_name
                sheet["days"][selected_day] = {
                    "color_name": color_name,
                    "color_hex": COLORS[color_name],
                }
                # Automatically move to the next uncolored valid day.
                next_day = None
                for d in range(selected_day + 1, N_DAYS + 1):
                    if actual_date(sheet, d) is not None and d not in sheet["days"]:
                        next_day = d
                        break
                if next_day is None:
                    for d in range(1, selected_day):
                        if actual_date(sheet, d) is not None and d not in sheet["days"]:
                            next_day = d
                            break
                if next_day is not None:
                    st.session_state.selected_day = next_day
                st.session_state.last_click_signature = None
                st.session_state.map_version += 1
                st.rerun()

    st.markdown("#### Saved dates")
    saved = []
    for d in sorted(sheet["days"]):
        ad = actual_date(sheet, d)
        if ad:
            saved.append({
                "Date": ad.strftime("%d-%b-%Y"),
                "Day": d,
                "Color": sheet["days"][d]["color_name"],
            })
    if saved:
        st.dataframe(pd.DataFrame(saved), hide_index=True, width="stretch")
    else:
        st.caption("No dates colored yet.")

    if st.button("Clear selected date", width="stretch"):
        sheet["days"].pop(selected_day, None)
        st.session_state.last_click_signature = None
        st.session_state.map_version += 1
        st.rerun()

    if st.button("Clear this sheet", width="stretch"):
        sheet["days"] = {}
        st.session_state.selected_day = 1
        st.session_state.last_click_signature = None
        st.session_state.map_version += 1
        st.rerun()

with preview:
    st.subheader("Sheet preview")
    fig = make_map(sheet)
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"date_map_{idx}_{st.session_state.map_version}",
        on_select="rerun",
        selection_mode=["points"],
    )

    if event is not None:
        points = getattr(getattr(event, "selection", None), "points", [])
        if points:
            clicked_day = None
            for point in points:
                cd = point.get("customdata") if isinstance(point, dict) else None
                if cd is not None:
                    try:
                        clicked_day = int(cd[0] if isinstance(cd, (list, tuple)) else cd)
                        break
                    except (TypeError, ValueError):
                        pass
            if clicked_day is not None and 1 <= clicked_day <= N_DAYS:
                sig = (idx, clicked_day)
                if sig != st.session_state.last_click_signature:
                    st.session_state.selected_day = clicked_day
                    st.session_state.last_click_signature = sig
                    st.session_state.map_version += 1
                    st.rerun()

    selected = st.session_state.selected_day
    existing = sheet["days"].get(selected)
    if existing:
        st.success(f"Date {selected} is already {existing['color_name']}. Choose another color to change it.")
    else:
        st.caption(f"Date {selected} is not colored yet.")

# ============================================================
# FINAL COLLAGE
# ============================================================
st.divider()
st.subheader("Final 10-sheet collage")
st.caption("All 10 sheets are preserved separately and combined into one final 4 + 4 + 2 collage.")

if st.button("Generate / Refresh 10-sheet collage", type="primary", width="stretch"):
    st.session_state.show_collage = True

if st.session_state.show_collage:
    images = [render_sheet(s) for s in st.session_state.sheets]
    final = collage(images, columns=4)
    st.image(final, width="stretch")
    st.download_button(
        "Download final collage PNG",
        data=png_bytes(final),
        file_name="solar_date_10_sheet_collage.png",
        mime="image/png",
        width="stretch",
    )

# Backup mapping
rows = []
for i, s in enumerate(st.session_state.sheets, start=1):
    for day in range(1, N_DAYS + 1):
        ad = actual_date(s, day)
        v = s["days"].get(day)
        rows.append({
            "Sheet": i,
            "Header": s["name"],
            "Month": s["header_date"].strftime("%b-%Y"),
            "Day": day,
            "Date": ad.isoformat() if ad else "",
            "Color": v["color_name"] if v else "",
            "MAL Capacity (MW)": s["mal"],
            "Total Capacity (MW)": s["total"],
        })
backup = pd.DataFrame(rows)
st.download_button(
    "Download date/color mapping CSV",
    backup.to_csv(index=False),
    "solar_date_color_mapping.csv",
    "text/csv",
)
