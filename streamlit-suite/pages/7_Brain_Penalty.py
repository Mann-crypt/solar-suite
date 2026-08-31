import io
import sqlite3
from datetime import date
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Solar Date Coloring",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIG
# ============================================================

N_SHEETS = 10
N_DAYS = 31

DB_PATH = Path("solar_date_coloring.db")

COLORS = {
    "Red": "#F26B4F",
    "Green": "#35C98B",
    "Yellow": "#FFF36A",
    "Blue": "#2F80ED",
}

DEFAULT_FILL = "#F26B4F"


# ============================================================
# DATE POSITIONS
# ============================================================

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

Y = [
    145,
    235,
    325,
    415,
    505,
    595,
    685,
    775,
    865,
    955,
]

FOUR_X = [
    (145, 300),
    (300, 455),
    (545, 700),
    (700, 855),
]


def make_polygons():

    polys = {}

    for ri, row in enumerate(ROWS):

        top = Y[ri]
        bottom = Y[ri + 1]

        if len(row) == 2:

            xs = [
                (320, 470),
                (530, 680),
            ]

        elif len(row) == 3:

            xs = [
                (245, 395),
                (395, 545),
                (545, 695),
            ]

        else:

            xs = FOUR_X

        for i, day in enumerate(row):

            x1, x2 = xs[i]

            slant = (
                8
                if ri < 2
                else 4
            )

            polys[day] = [
                (x1 + slant, top),
                (x2 - slant, top + 2),
                (x2, bottom - 3),
                (x1, bottom),
            ]

    return polys


POLYS = make_polygons()


def center(points):

    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


# ============================================================
# DATABASE
# ============================================================

@st.cache_resource
def get_database():

    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sheets (
            sheet_no INTEGER PRIMARY KEY,
            name TEXT DEFAULT '',
            header_date TEXT DEFAULT '',
            mal TEXT DEFAULT '',
            total TEXT DEFAULT ''
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS date_colors (
            sheet_no INTEGER NOT NULL,
            day INTEGER NOT NULL,
            color TEXT NOT NULL,

            PRIMARY KEY (
                sheet_no,
                day
            )
        )
        """
    )

    # Ensure all 10 sheets exist
    for sheet_no in range(
        1,
        N_SHEETS + 1,
    ):

        conn.execute(
            """
            INSERT OR IGNORE INTO sheets (
                sheet_no
            )
            VALUES (?)
            """,
            (sheet_no,),
        )

    conn.commit()

    return conn


DB = get_database()


# ============================================================
# DATABASE LOAD
# ============================================================

def load_sheet_from_database(
    sheet_no,
):

    cursor = DB.cursor()

    cursor.execute(
        """
        SELECT
            name,
            header_date,
            mal,
            total
        FROM sheets
        WHERE sheet_no = ?
        """,
        (sheet_no,),
    )

    row = cursor.fetchone()

    if row is None:

        header_date = date.today().replace(
            day=1
        )

        sheet = {
            "name": "",
            "header_date": header_date,
            "mal": "",
            "total": "",
            "days": {},
        }

    else:

        name = row[0] or ""
        raw_date = row[1] or ""
        mal = row[2] or ""
        total = row[3] or ""

        try:

            header_date = date.fromisoformat(
                raw_date
            ).replace(day=1)

        except Exception:

            header_date = date.today().replace(
                day=1
            )

        sheet = {
            "name": name,
            "header_date": header_date,
            "mal": mal,
            "total": total,
            "days": {},
        }

    cursor.execute(
        """
        SELECT
            day,
            color
        FROM date_colors
        WHERE sheet_no = ?
        ORDER BY day
        """,
        (sheet_no,),
    )

    for day, color in cursor.fetchall():

        if color in COLORS:

            sheet["days"][int(day)] = {
                "color_name": color,
                "color_hex": COLORS[color],
            }

    return sheet


# ============================================================
# DATABASE SAVE DETAILS
# ============================================================

def save_sheet_details(
    sheet_no,
    sheet,
):

    DB.execute(
        """
        UPDATE sheets
        SET
            name = ?,
            header_date = ?,
            mal = ?,
            total = ?
        WHERE sheet_no = ?
        """,
        (
            sheet["name"],
            sheet["header_date"].isoformat(),
            sheet["mal"],
            sheet["total"],
            sheet_no,
        ),
    )

    DB.commit()


# ============================================================
# DATABASE SAVE COLOR
# ============================================================

def save_date_color(
    sheet_no,
    day,
    color_name,
):

    if color_name not in COLORS:
        return

    DB.execute(
        """
        INSERT OR REPLACE INTO date_colors (
            sheet_no,
            day,
            color
        )
        VALUES (?, ?, ?)
        """,
        (
            sheet_no,
            day,
            color_name,
        ),
    )

    DB.commit()


# ============================================================
# DATABASE DELETE COLOR
# ============================================================

def delete_date_color(
    sheet_no,
    day,
):

    DB.execute(
        """
        DELETE FROM date_colors
        WHERE
            sheet_no = ?
            AND day = ?
        """,
        (
            sheet_no,
            day,
        ),
    )

    DB.commit()


# ============================================================
# DATABASE CLEAR SHEET
# ============================================================

def clear_sheet_database(
    sheet_no,
):

    DB.execute(
        """
        DELETE FROM date_colors
        WHERE sheet_no = ?
        """,
        (sheet_no,),
    )

    DB.commit()


# ============================================================
# MONTH DAYS
# ============================================================

def days_in_month(
    month_date,
):

    return calendar.monthrange(
        month_date.year,
        month_date.month,
    )[1]


def actual_date(
    sheet,
    day,
):

    try:

        return date(
            sheet["header_date"].year,
            sheet["header_date"].month,
            day,
        )

    except ValueError:

        return None


# ============================================================
# SESSION STATE
# ============================================================

if "active_sheet" not in st.session_state:

    st.session_state.active_sheet = 0


if "selected_day" not in st.session_state:

    st.session_state.selected_day = min(
        date.today().day,
        31,
    )


if "selected_color" not in st.session_state:

    st.session_state.selected_color = "Red"


if "map_version" not in st.session_state:

    st.session_state.map_version = 0


if "last_click_signature" not in st.session_state:

    st.session_state.last_click_signature = None


if "show_collage" not in st.session_state:

    st.session_state.show_collage = False


# ============================================================
# LOAD ALL SHEETS FROM DATABASE
#
# This is important:
# session_state is NOT the permanent storage.
# Database is the source of truth.
# ============================================================

if (
    "sheets_loaded"
    not in st.session_state
):

    st.session_state.sheets = [
        load_sheet_from_database(
            i
        )
        for i in range(
            1,
            N_SHEETS + 1,
        )
    ]

    st.session_state.sheets_loaded = True


def current_sheet():

    return st.session_state.sheets[
        st.session_state.active_sheet
    ]


# ============================================================
# VISUAL CSS
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BRAIN OUTLINE
# ============================================================

def brain_outline():

    return [
        (125, 150),
        (185, 110),
        (275, 82),
        (380, 75),
        (450, 92),
        (500, 120),

        (550, 92),
        (620, 75),
        (725, 82),
        (815, 110),
        (875, 150),

        (905, 225),
        (920, 330),
        (918, 460),
        (925, 600),
        (918, 740),
        (900, 865),

        (865, 960),
        (810, 1030),
        (725, 1085),
        (630, 1110),
        (555, 1115),

        (500, 1085),

        (445, 1115),
        (370, 1110),
        (275, 1085),
        (190, 1030),
        (135, 960),

        (100, 865),
        (82, 740),
        (75, 600),
        (82, 460),
        (80, 330),
        (95, 225),
    ]


# ============================================================
# BRAIN FOLDS
# ============================================================

def add_brain_folds(
    fig,
):

    folds = [
        # Left upper
        [
            (160, 205),
            (220, 170),
            (300, 180),
            (360, 150),
        ],
        [
            (125, 290),
            (205, 250),
            (280, 270),
            (350, 225),
        ],
        [
            (115, 390),
            (190, 350),
            (260, 370),
            (330, 320),
        ],
        [
            (105, 505),
            (180, 455),
            (255, 485),
            (330, 425),
        ],
        [
            (105, 630),
            (185, 575),
            (260, 610),
            (335, 550),
        ],
        [
            (115, 755),
            (190, 700),
            (265, 735),
            (335, 675),
        ],
        [
            (135, 875),
            (210, 820),
            (280, 855),
            (345, 800),
        ],
        [
            (175, 955),
            (240, 920),
            (300, 945),
            (365, 905),
        ],

        # Right upper
        [
            (840, 205),
            (780, 170),
            (700, 180),
            (640, 150),
        ],
        [
            (875, 290),
            (795, 250),
            (720, 270),
            (650, 225),
        ],
        [
            (885, 390),
            (810, 350),
            (740, 370),
            (670, 320),
        ],
        [
            (895, 505),
            (820, 455),
            (745, 485),
            (670, 425),
        ],
        [
            (895, 630),
            (815, 575),
            (740, 610),
            (665, 550),
        ],
        [
            (885, 755),
            (810, 700),
            (735, 735),
            (665, 675),
        ],
        [
            (865, 875),
            (790, 820),
            (720, 855),
            (655, 800),
        ],
        [
            (825, 955),
            (760, 920),
            (700, 945),
            (635, 905),
        ],
    ]

    for fold in folds:

        fig.add_trace(
            go.Scatter(
                x=[
                    p[0]
                    for p in fold
                ],
                y=[
                    p[1]
                    for p in fold
                ],
                mode="lines",
                line=dict(
                    color="#9A9A9A",
                    width=2,
                    shape="spline",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )


# ============================================================
# PLOTLY BRAIN MAP
# ============================================================

def make_map(
    sheet,
):

    fig = go.Figure()

    # --------------------------------------------------------
    # Brain fill behind dates
    # --------------------------------------------------------

    outline = brain_outline()

    fig.add_trace(
        go.Scatter(
            x=[
                p[0]
                for p in outline
            ]
            + [outline[0][0]],
            y=[
                p[1]
                for p in outline
            ]
            + [outline[0][1]],
            mode="lines",
            fill="toself",
            fillcolor="#FFFDF8",
            line=dict(
                color="black",
                width=5,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Date blocks
    # --------------------------------------------------------

    valid_days = days_in_month(
        sheet["header_date"]
    )

    for day in range(
        1,
        N_DAYS + 1,
    ):

        pts = POLYS[day]

        # Dates that don't exist in the selected month
        # remain visually muted.
        valid = day <= valid_days

        assigned = sheet[
            "days"
        ].get(day)

        if assigned:

            fill = assigned[
                "color_hex"
            ]

        elif valid:

            fill = DEFAULT_FILL

        else:

            fill = "#E5E5E5"

        xs = [
            p[0]
            for p in pts
        ] + [pts[0][0]]

        ys = [
            p[1]
            for p in pts
        ] + [pts[0][1]]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(
                    color="#222222",
                    width=2,
                ),
                customdata=[
                    day
                ]
                * len(xs),
                hovertemplate=(
                    f"<b>Date {day}</b>"
                    "<br>"
                    f"{actual_date(sheet, day).strftime('%d-%b-%Y') if valid else 'Not in selected month'}"
                    "<extra></extra>"
                ),
                hoverinfo="text",
                showlegend=False,
                opacity=0.96 if valid else 0.35,
            )
        )

        # ----------------------------------------------------
        # Date number
        # ----------------------------------------------------

        cx, cy = center(
            pts
        )

        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                text=[
                    str(day)
                ],
                textposition="middle center",
                marker=dict(
                    size=43,
                    color=(
                        "rgba(255,255,255,0.01)"
                        if valid
                        else "rgba(255,255,255,0)"
                    ),
                    line=dict(
                        width=0
                    ),
                ),
                textfont=dict(
                    size=16,
                    color="#111111",
                    family="Arial",
                ),
                customdata=[
                    day
                ],
                hovertemplate=(
                    f"<b>Click Date {day}</b>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # Brain center line
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                500,
                500,
            ],
            y=[
                112,
                1090,
            ],
            mode="lines",
            line=dict(
                color="#222222",
                width=5,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Brain folds
    # --------------------------------------------------------

    add_brain_folds(
        fig
    )

    # --------------------------------------------------------
    # Outer border again
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                p[0]
                for p in outline
            ]
            + [outline[0][0]],
            y=[
                p[1]
                for p in outline
            ]
            + [outline[0][1]],
            mode="lines",
            line=dict(
                color="#111111",
                width=6,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_xaxes(
        visible=False,
        range=[
            40,
            960,
        ],
        fixedrange=True,
    )

    fig.update_yaxes(
        visible=False,
        range=[
            1150,
            45,
        ],
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        height=760,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        clickmode="event+select",
        dragmode=False,
        hovermode="closest",
        showlegend=False,
    )

    return fig


# ============================================================
# IMAGE FONT
# ============================================================

def load_font(
    size,
    bold=False,
):

    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans-Bold.ttf"
            if bold
            else
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans.ttf"
        ),
        (
            "C:/Windows/Fonts/arialbd.ttf"
            if bold
            else
            "C:/Windows/Fonts/arial.ttf"
        ),
    ]

    for path in candidates:

        try:

            return ImageFont.truetype(
                path,
                size,
            )

        except OSError:

            pass

    return ImageFont.load_default()


# ============================================================
# STATIC PRINTABLE SHEET
# ============================================================

def render_sheet(
    sheet,
):

    W = 1000
    H = 1300

    img = Image.new(
        "RGB",
        (
            W,
            H,
        ),
        "white",
    )

    d = ImageDraw.Draw(
        img
    )

    title_font = load_font(
        34,
        True,
    )

    month_font = load_font(
        31,
        True,
    )

    small_font = load_font(
        20,
    )

    day_font = load_font(
        22,
        True,
    )

    # --------------------------------------------------------
    # Outer page border
    # --------------------------------------------------------

    d.rounded_rectangle(
        (
            18,
            18,
            W - 18,
            H - 18,
        ),
        radius=8,
        outline="#111111",
        width=3,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    title = (
        sheet["name"]
        or "Solar - ____"
    )

    month = sheet[
        "header_date"
    ].strftime(
        "%B %Y"
    )

    d.text(
        (
            60,
            48,
        ),
        title,
        fill="#111111",
        font=title_font,
    )

    d.line(
        (
            55,
            94,
            340,
            94,
        ),
        fill="#111111",
        width=2,
    )

    month_box = d.textbbox(
        (
            0,
            0,
        ),
        month,
        font=month_font,
    )

    month_width = (
        month_box[2]
        - month_box[0]
    )

    d.text(
        (
            W
            - 60
            - month_width,
            50,
        ),
        month,
        fill="#111111",
        font=month_font,
    )

    # --------------------------------------------------------
    # Brain
    # --------------------------------------------------------

    outline = brain_outline()

    d.polygon(
        outline,
        fill="#FFFDF8",
    )

    # --------------------------------------------------------
    # Date blocks
    # --------------------------------------------------------

    valid_days = days_in_month(
        sheet["header_date"]
    )

    for day, pts in POLYS.items():

        valid = (
            day <= valid_days
        )

        assigned = sheet[
            "days"
        ].get(day)

        if assigned:

            fill = assigned[
                "color_hex"
            ]

        elif valid:

            fill = DEFAULT_FILL

        else:

            fill = "#E6E6E6"

        d.polygon(
            pts,
            fill=fill,
            outline="#222222",
        )

        cx, cy = center(
            pts
        )

        text = str(day)

        box = d.textbbox(
            (
                0,
                0,
            ),
            text,
            font=day_font,
        )

        tw = (
            box[2]
            - box[0]
        )

        th = (
            box[3]
            - box[1]
        )

        d.text(
            (
                cx - tw / 2,
                cy - th / 2,
            ),
            text,
            fill="#111111",
            font=day_font,
        )

    # --------------------------------------------------------
    # Brain folds
    # --------------------------------------------------------

    fold_sets = [
        [
            (160, 205),
            (220, 170),
            (300, 180),
            (360, 150),
        ],
        [
            (125, 290),
            (205, 250),
            (280, 270),
            (350, 225),
        ],
        [
            (115, 390),
            (190, 350),
            (260, 370),
            (330, 320),
        ],
        [
            (105, 505),
            (180, 455),
            (255, 485),
            (330, 425),
        ],
        [
            (105, 630),
            (185, 575),
            (260, 610),
            (335, 550),
        ],
        [
            (115, 755),
            (190, 700),
            (265, 735),
            (335, 675),
        ],
        [
            (135, 875),
            (210, 820),
            (280, 855),
            (345, 800),
        ],
        [
            (175, 955),
            (240, 920),
            (300, 945),
            (365, 905),
        ],

        [
            (840, 205),
            (780, 170),
            (700, 180),
            (640, 150),
        ],
        [
            (875, 290),
            (795, 250),
            (720, 270),
            (650, 225),
        ],
        [
            (885, 390),
            (810, 350),
            (740, 370),
            (670, 320),
        ],
        [
            (895, 505),
            (820, 455),
            (745, 485),
            (670, 425),
        ],
        [
            (895, 630),
            (815, 575),
            (740, 610),
            (665, 550),
        ],
        [
            (885, 755),
            (810, 700),
            (735, 735),
            (665, 675),
        ],
        [
            (865, 875),
            (790, 820),
            (720, 855),
            (655, 800),
        ],
        [
            (825, 955),
            (760, 920),
            (700, 945),
            (635, 905),
        ],
    ]

    for fold in fold_sets:

        d.line(
            fold,
            fill="#A0A0A0",
            width=2,
            joint="curve",
        )

    # --------------------------------------------------------
    # Center split
    # --------------------------------------------------------

    d.line(
        (
            500,
            112,
            500,
            1090,
        ),
        fill="#111111",
        width=7,
    )

    # --------------------------------------------------------
    # Outer brain outline
    # --------------------------------------------------------

    d.line(
        outline + [outline[0]],
        fill="#111111",
        width=6,
        joint="curve",
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    footer_y = 1160

    mal = (
        sheet["mal"]
        or "________"
    )

    total = (
        sheet["total"]
        or "________"
    )

    d.text(
        (
            60,
            footer_y,
        ),
        f"MAL Capacity (MW): {mal}",
        fill="#111111",
        font=small_font,
    )

    d.text(
        (
            600,
            footer_y,
        ),
        "No. of PSS: ________",
        fill="#111111",
        font=small_font,
    )

    d.text(
        (
            60,
            footer_y + 45,
        ),
        f"Total Capacity (MW): {total}",
        fill="#111111",
        font=small_font,
    )

    return img


# ============================================================
# COLLAGE
# ============================================================

def make_collage(
    images,
    columns=4,
):

    gap = 20

    thumb_w = 520
    thumb_h = 676

    rows = int(
        np.ceil(
            len(images)
            / columns
        )
    )

    width = (
        columns
        * thumb_w
        + (columns + 1)
        * gap
    )

    height = (
        rows
        * thumb_h
        + (rows + 1)
        * gap
    )

    canvas = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "#E8E8E8",
    )

    for i, image in enumerate(
        images
    ):

        copy = image.copy()

        copy.thumbnail(
            (
                thumb_w,
                thumb_h,
            ),
            Image.Resampling.LANCZOS,
        )

        x = (
            gap
            + (
                i
                % columns
            )
            * (
                thumb_w
                + gap
            )
            + (
                thumb_w
                - copy.width
            )
            // 2
        )

        y = (
            gap
            + (
                i
                // columns
            )
            * (
                thumb_h
                + gap
            )
            + (
                thumb_h
                - copy.height
            )
            // 2
        )

        canvas.paste(
            copy,
            (
                x,
                y,
            ),
        )

    return canvas


# ============================================================
# PNG
# ============================================================

def image_bytes(
    image,
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.title(
    "☀️ Solar Date Coloring"
)

st.caption(
    "Click a date, choose Red, Green, Yellow or Blue. "
    "Date colors are saved immediately."
)


# ============================================================
# SHEET NAVIGATION
# ============================================================

st.subheader(
    "Sheets"
)

sheet_buttons = st.columns(
    N_SHEETS
)

for i, col in enumerate(
    sheet_buttons
):

    with col:

        current = (
            i
            == st.session_state.active_sheet
        )

        if current:

            label = f"🟦 {i + 1}"

        else:

            label = f"{i + 1}"

        if st.button(
            label,
            key=f"sheet_nav_{i}",
            use_container_width=True,
        ):

            st.session_state.active_sheet = i

            st.session_state.selected_day = min(
                date.today().day,
                31,
            )

            st.session_state.last_click_signature = None

            st.session_state.map_version += 1

            st.rerun()


# ============================================================
# CURRENT SHEET
# ============================================================

sheet_index = (
    st.session_state.active_sheet
)

sheet = current_sheet()


# ============================================================
# MAIN LAYOUT
# ============================================================

st.divider()

controls, preview = st.columns(
    [
        0.75,
        1.65,
    ],
    gap="large",
)


# ============================================================
# CONTROLS
# ============================================================

with controls:

    st.subheader(
        f"Sheet {sheet_index + 1}"
    )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    sheet["name"] = st.text_input(
        "Write on left header",
        value=sheet["name"],
        key=f"name_input_{sheet_index}",
        placeholder="Solar - G1",
    )

    sheet["header_date"] = st.date_input(
        "Header month",
        value=sheet["header_date"],
        key=f"date_input_{sheet_index}",
    ).replace(
        day=1
    )

    sheet["mal"] = st.text_input(
        "MAL Capacity (MW)",
        value=sheet["mal"],
        key=f"mal_input_{sheet_index}",
    )

    sheet["total"] = st.text_input(
        "Total Capacity (MW)",
        value=sheet["total"],
        key=f"total_input_{sheet_index}",
    )

    # --------------------------------------------------------
    # DATABASE SAVE
    # --------------------------------------------------------

    st.markdown(
        "### Database"
    )

    st.caption(
        "Save the header, month and capacity details permanently."
    )

    if st.button(
        "💾 Save Details to Database",
        type="primary",
        use_container_width=True,
        key=f"save_details_{sheet_index}",
    ):

        save_sheet_details(
            sheet_index + 1,
            sheet,
        )

        # Immediately reload from DB
        st.session_state.sheets[
            sheet_index
        ] = load_sheet_from_database(
            sheet_index + 1
        )

        st.success(
            "Details saved successfully."
        )

        st.rerun()

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    st.markdown(
        "### Color Dates"
    )

    st.info(
        "Click a date on the brain preview, "
        "then choose its color."
    )

    selected_day = (
        st.session_state.selected_day
    )

    selected_actual_date = actual_date(
        sheet,
        selected_day,
    )

    if selected_actual_date:

        st.write(
            f"**Selected:** "
            f"{selected_actual_date.strftime('%d-%b-%Y')}"
        )

    else:

        st.write(
            f"**Selected day:** {selected_day}"
        )

    # --------------------------------------------------------
    # COLOR BUTTONS
    # --------------------------------------------------------

    color_cols = st.columns(
        4
    )

    for color_name, col in zip(
        COLORS,
        color_cols,
    ):

        with col:

            if st.button(
                color_name,
                key=(
                    f"color_button_"
                    f"{sheet_index}_"
                    f"{color_name}"
                ),
                use_container_width=True,
            ):

                if (
                    selected_actual_date
                    is not None
                ):

                    save_date_color(
                        sheet_index + 1,
                        selected_day,
                        color_name,
                    )

                    sheet["days"][
                        selected_day
                    ] = {
                        "color_name": color_name,
                        "color_hex": COLORS[
                            color_name
                        ],
                    }

                    st.session_state.selected_color = (
                        color_name
                    )

                    # ----------------------------------------
                    # AUTO NEXT UNCOLORED DATE
                    # ----------------------------------------

                    valid_days = days_in_month(
                        sheet[
                            "header_date"
                        ]
                    )

                    next_day = None

                    for candidate in range(
                        selected_day + 1,
                        valid_days + 1,
                    ):

                        if (
                            candidate
                            not in sheet[
                                "days"
                            ]
                        ):

                            next_day = candidate

                            break

                    if next_day is None:

                        for candidate in range(
                            1,
                            valid_days + 1,
                        ):

                            if (
                                candidate
                                not in sheet[
                                    "days"
                                ]
                            ):

                                next_day = candidate

                                break

                    if next_day is not None:

                        st.session_state.selected_day = (
                            next_day
                        )

                    st.session_state.last_click_signature = None

                    st.session_state.map_version += 1

                    st.rerun()

    # --------------------------------------------------------
    # CURRENT COLOR
    # --------------------------------------------------------

    existing = sheet[
        "days"
    ].get(
        selected_day
    )

    if existing:

        st.success(
            f"Date {selected_day} is "
            f"{existing['color_name']}."
        )

    else:

        st.caption(
            f"Date {selected_day} is not colored."
        )

    # --------------------------------------------------------
    # DELETE SELECTED
    # --------------------------------------------------------

    if st.button(
        "🗑 Clear Selected Date",
        use_container_width=True,
        key=f"clear_date_{sheet_index}",
    ):

        delete_date_color(
            sheet_index + 1,
            selected_day,
        )

        sheet[
            "days"
        ].pop(
            selected_day,
            None,
        )

        st.session_state.map_version += 1

        st.rerun()

    # --------------------------------------------------------
    # CLEAR SHEET
    # --------------------------------------------------------

    if st.button(
        "🗑 Clear All Dates on This Sheet",
        use_container_width=True,
        key=f"clear_sheet_{sheet_index}",
    ):

        clear_sheet_database(
            sheet_index + 1
        )

        sheet[
            "days"
        ] = {}

        st.session_state.selected_day = 1

        st.session_state.map_version += 1

        st.rerun()


# ============================================================
# PREVIEW
# ============================================================

with preview:

    st.subheader(
        "Sheet Preview"
    )

    # --------------------------------------------------------
    # PRINTABLE HEADER
    # --------------------------------------------------------

    head_left, head_right = st.columns(
        [
            3,
            1,
        ]
    )

    with head_left:

        st.markdown(
            f"### {sheet['name'] or 'Solar - ____'}"
        )

    with head_right:

        st.markdown(
            f"### {sheet['header_date'].strftime('%B %Y')}"
        )

    # --------------------------------------------------------
    # BRAIN
    # --------------------------------------------------------

    fig = make_map(
        sheet
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=(
            f"brain_map_"
            f"{sheet_index}_"
            f"{st.session_state.map_version}"
        ),
        on_select="rerun",
        selection_mode=[
            "points"
        ],
    )

    # --------------------------------------------------------
    # HANDLE CLICK
    # --------------------------------------------------------

    if event is not None:

        points = getattr(
            getattr(
                event,
                "selection",
                None,
            ),
            "points",
            [],
        )

        if points:

            clicked_day = None

            for point in points:

                customdata = (
                    point.get(
                        "customdata"
                    )
                    if isinstance(
                        point,
                        dict,
                    )
                    else None
                )

                if customdata is None:
                    continue

                try:

                    if isinstance(
                        customdata,
                        (
                            list,
                            tuple,
                        ),
                    ):

                        clicked_day = int(
                            customdata[0]
                        )

                    else:

                        clicked_day = int(
                            customdata
                        )

                    break

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

            if (
                clicked_day is not None
                and 1 <= clicked_day <= N_DAYS
            ):

                click_signature = (
                    sheet_index,
                    clicked_day,
                )

                if (
                    click_signature
                    != st.session_state.last_click_signature
                ):

                    if actual_date(
                        sheet,
                        clicked_day,
                    ):

                        st.session_state.selected_day = (
                            clicked_day
                        )

                    st.session_state.last_click_signature = (
                        click_signature
                    )

                    st.session_state.map_version += 1

                    st.rerun()

    # --------------------------------------------------------
    # SELECTED DATE CARD
    # --------------------------------------------------------

    selected_day = (
        st.session_state.selected_day
    )

    selected_date = actual_date(
        sheet,
        selected_day,
    )

    if selected_date:

        selected_existing = sheet[
            "days"
        ].get(
            selected_day
        )

        if selected_existing:

            st.success(
                f"Selected Date: "
                f"{selected_date.strftime('%d-%b-%Y')}  •  "
                f"{selected_existing['color_name']}"
            )

        else:

            st.info(
                f"Selected Date: "
                f"{selected_date.strftime('%d-%b-%Y')}  •  "
                f"Not colored"
            )


# ============================================================
# COLOR LEGEND
# ============================================================

st.divider()

st.subheader(
    "Color Legend"
)

legend = st.columns(
    4
)

for col, (
    color_name,
    color_hex,
) in zip(
    legend,
    COLORS.items(),
):

    with col:

        st.markdown(
            f"""
            <div style="
                border:1px solid #D0D0D0;
                border-radius:8px;
                padding:10px;
                text-align:center;
                background:{color_hex};
                color:#111111;
                font-weight:600;
            ">
                {color_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SAVED DATE TABLE
# ============================================================

st.divider()

st.subheader(
    f"Saved Dates - Sheet {sheet_index + 1}"
)

saved_rows = []

for day in sorted(
    sheet["days"]
):

    actual = actual_date(
        sheet,
        day,
    )

    if actual:

        saved_rows.append(
            {
                "Date": actual.strftime(
                    "%d-%b-%Y"
                ),
                "Day": day,
                "Color": sheet[
                    "days"
                ][day][
                    "color_name"
                ],
            }
        )

if saved_rows:

    st.dataframe(
        pd.DataFrame(
            saved_rows
        ),
        hide_index=True,
        use_container_width=True,
    )

else:

    st.caption(
        "No dates colored yet."
    )


# ============================================================
# FINAL 10-SHEET COLLAGE
# ============================================================

st.divider()

st.subheader(
    "Final 10-Sheet Collage"
)

st.caption(
    "The collage is generated from the saved sheet data."
)

if st.button(
    "🧠 Generate / Refresh 10-Sheet Collage",
    type="primary",
    use_container_width=True,
):

    st.session_state.show_collage = True


if st.session_state.show_collage:

    # Always load the latest database data
    latest_sheets = [
        load_sheet_from_database(
            i
        )
        for i in range(
            1,
            N_SHEETS + 1,
        )
    ]

    images = [
        render_sheet(
            s
        )
        for s in latest_sheets
    ]

    final_collage = make_collage(
        images,
        columns=4,
    )

    st.image(
        final_collage,
        use_container_width=True,
    )

    st.download_button(
        "📥 Download 10-Sheet Collage",
        data=image_bytes(
            final_collage
        ),
        file_name=(
            "solar_date_10_sheet_collage.png"
        ),
        mime="image/png",
        use_container_width=True,
    )


# ============================================================
# COMPLETE DATABASE BACKUP
# ============================================================

st.divider()

st.subheader(
    "Database Backup"
)

all_rows = []

for sheet_no in range(
    1,
    N_SHEETS + 1,
):

    db_sheet = load_sheet_from_database(
        sheet_no
    )

    for day in range(
        1,
        N_DAYS + 1,
    ):

        actual = actual_date(
            db_sheet,
            day,
        )

        assignment = db_sheet[
            "days"
        ].get(
            day
        )

        all_rows.append(
            {
                "Sheet": sheet_no,
                "Header": db_sheet[
                    "name"
                ],
                "Month": db_sheet[
                    "header_date"
                ].strftime(
                    "%B %Y"
                ),
                "Day": day,
                "Date": (
                    actual.isoformat()
                    if actual
                    else ""
                ),
                "Color": (
                    assignment[
                        "color_name"
                    ]
                    if assignment
                    else ""
                ),
                "MAL Capacity (MW)": db_sheet[
                    "mal"
                ],
                "Total Capacity (MW)": db_sheet[
                    "total"
                ],
            }
        )


backup_df = pd.DataFrame(
    all_rows
)

st.download_button(
    "📥 Download Database Backup CSV",
    data=backup_df.to_csv(
        index=False
    ),
    file_name=(
        "solar_date_coloring_database_backup.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)
