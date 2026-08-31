import io
import sqlite3
import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
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

BORDER = "#111111"
PAGE_BG = "#FFFFFF"
BRAIN_BG = "#FFFDF7"


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

if "map_version" not in st.session_state:
    st.session_state.map_version = 0

if "last_click_signature" not in st.session_state:
    st.session_state.last_click_signature = None

if "show_collage" not in st.session_state:
    st.session_state.show_collage = False

if "sheets_loaded" not in st.session_state:
    st.session_state.sheets_loaded = False


# ============================================================
# CSS
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
            PRIMARY KEY (sheet_no, day)
        )
        """
    )

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

def load_sheet_from_database(sheet_no):

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
# INITIAL LOAD
# ============================================================

if not st.session_state.sheets_loaded:

    st.session_state.sheets = [
        load_sheet_from_database(i)
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
# DATABASE SAVE
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


def save_date_color(
    sheet_no,
    day,
    color_name,
):

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
# DATE HELPERS
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
# REFERENCE-LIKE BRAIN GEOMETRY
# ============================================================
#
# The important difference from the previous version:
#
# 1. Date cells are generated as adjoining regions.
# 2. There are no intentional gaps between cells.
# 3. Internal borders are drawn once.
# 4. The outer brain boundary is drawn separately.
#
# This makes the colors appear continuous like the reference.
# ============================================================


# ------------------------------------------------------------
# Brain outline
# ------------------------------------------------------------

def brain_outline():

    return [
        (110, 180),
        (145, 135),
        (205, 105),
        (275, 90),
        (345, 88),
        (410, 100),
        (455, 125),
        (500, 145),

        (545, 125),
        (590, 100),
        (655, 88),
        (725, 90),
        (795, 105),
        (855, 135),
        (890, 180),

        (910, 245),
        (915, 315),
        (905, 390),
        (920, 465),
        (915, 545),
        (925, 620),
        (915, 700),
        (920, 775),
        (900, 850),
        (875, 925),
        (830, 990),
        (770, 1040),
        (700, 1075),
        (625, 1095),
        (555, 1100),

        (500, 1070),

        (445, 1100),
        (375, 1095),
        (300, 1075),
        (230, 1040),
        (170, 990),
        (125, 925),
        (100, 850),
        (80, 775),
        (85, 700),
        (75, 620),
        (85, 545),
        (80, 465),
        (95, 390),
        (85, 315),
        (90, 245),
    ]


# ============================================================
# DATE CELL GEOMETRY
# ============================================================
#
# Instead of independent floating polygons, the cells form
# continuous horizontal bands.
# ============================================================


def make_date_cells():

    cells = {}

    # --------------------------------------------------------
    # Top two blocks
    # --------------------------------------------------------

    cells[1] = [
        (300, 145),
        (455, 145),
        (465, 230),
        (295, 230),
    ]

    cells[2] = [
        (545, 145),
        (700, 145),
        (705, 230),
        (535, 230),
    ]

    # --------------------------------------------------------
    # Second row
    # --------------------------------------------------------

    cells[3] = [
        (270, 230),
        (385, 230),
        (385, 315),
        (260, 315),
    ]

    cells[4] = [
        (385, 230),
        (500, 230),
        (500, 315),
        (385, 315),
    ]

    cells[5] = [
        (500, 230),
        (615, 230),
        (615, 315),
        (500, 315),
    ]

    cells[6] = [
        (615, 230),
        (740, 230),
        (740, 315),
        (615, 315),
    ]

    # --------------------------------------------------------
    # Main body
    # --------------------------------------------------------

    row_days = [
        [7, 8, 9, 10],
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22],
        [23, 24, 25, 26],
        [27, 28, 29, 30],
        [31],
    ]

    row_tops = [
        315,
        410,
        505,
        600,
        695,
        790,
        885,
    ]

    row_bottoms = [
        410,
        505,
        600,
        695,
        790,
        885,
        980,
    ]

    for row_index, days in enumerate(
        row_days
    ):

        top = row_tops[
            row_index
        ]

        bottom = row_bottoms[
            row_index
        ]

        count = len(days)

        if count == 1:

            x_ranges = [
                (
                    365,
                    635,
                )
            ]

        else:

            left = 155
            right = 845

            width = (
                right - left
            ) / count

            x_ranges = []

            for i in range(count):

                x1 = (
                    left
                    + i * width
                )

                x2 = (
                    left
                    + (i + 1) * width
                )

                x_ranges.append(
                    (
                        x1,
                        x2,
                    )
                )

        for day, (
            x1,
            x2,
        ) in zip(
            days,
            x_ranges,
        ):

            # Slight organic curvature on outer cells
            if x1 < 200:
                top_x1 = x1 + 18
                bottom_x1 = x1
            else:
                top_x1 = x1
                bottom_x1 = x1

            if x2 > 800:
                top_x2 = x2 - 18
                bottom_x2 = x2
            else:
                top_x2 = x2
                bottom_x2 = x2

            cells[day] = [
                (
                    top_x1,
                    top,
                ),
                (
                    top_x2,
                    top,
                ),
                (
                    bottom_x2,
                    bottom,
                ),
                (
                    bottom_x1,
                    bottom,
                ),
            ]

    return cells


DATE_CELLS = make_date_cells()


# ============================================================
# SHARED BOUNDARY LINES
# ============================================================

def shared_boundaries():

    boundaries = []

    # --------------------------------------------------------
    # Horizontal boundaries
    # --------------------------------------------------------

    horizontal = [
        (
            230,
            270,
            740,
        ),
        (
            315,
            155,
            845,
        ),
        (
            410,
            155,
            845,
        ),
        (
            505,
            155,
            845,
        ),
        (
            600,
            155,
            845,
        ),
        (
            695,
            155,
            845,
        ),
        (
            790,
            155,
            845,
        ),
        (
            885,
            155,
            845,
        ),
    ]

    for y, x1, x2 in horizontal:

        boundaries.append(
            [
                (
                    x1,
                    y,
                ),
                (
                    x2,
                    y,
                ),
            ]
        )

    # --------------------------------------------------------
    # Vertical center
    # --------------------------------------------------------

    boundaries.append(
        [
            (
                500,
                145,
            ),
            (
                500,
                1070,
            ),
        ]
    )

    return boundaries


SHARED_BOUNDARIES = (
    shared_boundaries()
)


# ============================================================
# ORGANIC BRAIN FOLDS
# ============================================================

def brain_folds():

    return [

        # Left upper
        [
            (115, 245),
            (170, 205),
            (235, 220),
            (285, 180),
        ],

        [
            (100, 325),
            (155, 285),
            (220, 300),
            (265, 265),
        ],

        [
            (95, 415),
            (155, 375),
            (215, 395),
            (260, 350),
        ],

        [
            (90, 515),
            (150, 470),
            (215, 495),
            (255, 450),
        ],

        [
            (90, 620),
            (150, 570),
            (215, 605),
            (265, 550),
        ],

        [
            (95, 725),
            (155, 680),
            (220, 710),
            (270, 665),
        ],

        [
            (115, 835),
            (175, 785),
            (235, 820),
            (285, 770),
        ],

        [
            (150, 930),
            (205, 885),
            (260, 920),
            (320, 870),
        ],

        # Right upper
        [
            (885, 245),
            (830, 205),
            (765, 220),
            (715, 180),
        ],

        [
            (900, 325),
            (845, 285),
            (780, 300),
            (735, 265),
        ],

        [
            (905, 415),
            (845, 375),
            (785, 395),
            (740, 350),
        ],

        [
            (910, 515),
            (850, 470),
            (785, 495),
            (745, 450),
        ],

        [
            (910, 620),
            (850, 570),
            (785, 605),
            (735, 550),
        ],

        [
            (905, 725),
            (845, 680),
            (780, 710),
            (730, 665),
        ],

        [
            (885, 835),
            (825, 785),
            (765, 820),
            (715, 770),
        ],

        [
            (850, 930),
            (795, 885),
            (740, 920),
            (680, 870),
        ],
    ]


# ============================================================
# PLOTLY PREVIEW
# ============================================================

def make_map(sheet):

    fig = go.Figure()

    outline = brain_outline()

    # --------------------------------------------------------
    # Brain base
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
            fill="toself",
            fillcolor=BRAIN_BG,
            line=dict(
                color=BORDER,
                width=5,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    valid_days = days_in_month(
        sheet["header_date"]
    )

    # --------------------------------------------------------
    # Date cells
    # --------------------------------------------------------

    for day in range(
        1,
        N_DAYS + 1,
    ):

        if day not in DATE_CELLS:
            continue

        pts = DATE_CELLS[day]

        valid = (
            day <= valid_days
        )

        assignment = sheet[
            "days"
        ].get(day)

        if assignment:

            fill = assignment[
                "color_hex"
            ]

        elif valid:

            fill = DEFAULT_FILL

        else:

            fill = "#E6E6E6"

        xs = [
            p[0]
            for p in pts
        ] + [pts[0][0]]

        ys = [
            p[1]
            for p in pts
        ] + [pts[0][1]]

        actual = actual_date(
            sheet,
            day,
        )

        actual_text = (
            actual.strftime(
                "%d-%b-%Y"
            )
            if actual
            else "Invalid date"
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(
                    color="rgba(0,0,0,0)",
                    width=0,
                ),
                customdata=[
                    day
                ]
                * len(xs),
                hovertemplate=(
                    f"<b>Date {day}</b>"
                    f"<br>{actual_text}"
                    "<br><br>"
                    "Click to select"
                    "<extra></extra>"
                ),
                showlegend=False,
                opacity=(
                    1.0
                    if valid
                    else 0.35
                ),
            )
        )

        # --------------------------------------------
        # Number
        # --------------------------------------------

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
                    size=38,
                    color="rgba(255,255,255,0.01)",
                    line=dict(
                        width=0
                    ),
                ),
                textfont=dict(
                    size=17,
                    color="#111111",
                    family="Arial",
                ),
                customdata=[
                    day
                ],
                hovertemplate=(
                    f"<b>Select Date {day}</b>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # Internal shared borders
    # --------------------------------------------------------

    for boundary in SHARED_BOUNDARIES:

        fig.add_trace(
            go.Scatter(
                x=[
                    p[0]
                    for p in boundary
                ],
                y=[
                    p[1]
                    for p in boundary
                ],
                mode="lines",
                line=dict(
                    color=BORDER,
                    width=2,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # Brain folds
    # --------------------------------------------------------

    for fold in brain_folds():

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
                    color="#777777",
                    width=2,
                    shape="spline",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # Center division
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                500,
                500,
            ],
            y=[
                115,
                1085,
            ],
            mode="lines",
            line=dict(
                color=BORDER,
                width=5,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Outer border
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
                color=BORDER,
                width=6,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        visible=False,
        range=[
            45,
            955,
        ],
        fixedrange=True,
    )

    fig.update_yaxes(
        visible=False,
        range=[
            1140,
            55,
        ],
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        height=780,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        clickmode="event+select",
        dragmode=False,
        hovermode="closest",
        showlegend=False,
    )

    return fig


# ============================================================
# FONT
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
# PRINTABLE SHEET
# ============================================================

def render_sheet(sheet):

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
        21,
        True,
    )

    # --------------------------------------------------------
    # PAGE BORDER
    # --------------------------------------------------------

    d.rounded_rectangle(
        (
            18,
            18,
            W - 18,
            H - 18,
        ),
        radius=8,
        outline=BORDER,
        width=3,
    )

    # --------------------------------------------------------
    # HEADER
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
        fill=BORDER,
        font=title_font,
    )

    d.line(
        (
            55,
            94,
            340,
            94,
        ),
        fill=BORDER,
        width=2,
    )

    box = d.textbbox(
        (
            0,
            0,
        ),
        month,
        font=month_font,
    )

    month_width = (
        box[2]
        - box[0]
    )

    d.text(
        (
            W
            - 60
            - month_width,
            50,
        ),
        month,
        fill=BORDER,
        font=month_font,
    )

    # --------------------------------------------------------
    # BRAIN BASE
    # --------------------------------------------------------

    outline = brain_outline()

    d.polygon(
        outline,
        fill=BRAIN_BG,
    )

    valid_days = days_in_month(
        sheet["header_date"]
    )

    # --------------------------------------------------------
    # DATE CELLS
    # --------------------------------------------------------

    for day, pts in DATE_CELLS.items():

        valid = (
            day <= valid_days
        )

        assignment = sheet[
            "days"
        ].get(day)

        if assignment:

            fill = assignment[
                "color_hex"
            ]

        elif valid:

            fill = DEFAULT_FILL

        else:

            fill = "#E5E5E5"

        d.polygon(
            pts,
            fill=fill,
        )

    # --------------------------------------------------------
    # INTERNAL SHARED BORDERS
    # --------------------------------------------------------

    for boundary in SHARED_BOUNDARIES:

        d.line(
            boundary,
            fill=BORDER,
            width=2,
        )

    # --------------------------------------------------------
    # BRAIN FOLDS
    # --------------------------------------------------------

    for fold in brain_folds():

        d.line(
            fold,
            fill="#777777",
            width=2,
            joint="curve",
        )

    # --------------------------------------------------------
    # DATE NUMBERS
    # --------------------------------------------------------

    for day, pts in DATE_CELLS.items():

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
            fill=BORDER,
            font=day_font,
        )

    # --------------------------------------------------------
    # CENTER LINE
    # --------------------------------------------------------

    d.line(
        (
            500,
            115,
            500,
            1085,
        ),
        fill=BORDER,
        width=5,
    )

    # --------------------------------------------------------
    # OUTER BRAIN BORDER
    # --------------------------------------------------------

    d.line(
        outline + [outline[0]],
        fill=BORDER,
        width=6,
        joint="curve",
    )

    # --------------------------------------------------------
    # FOOTER
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
        fill=BORDER,
        font=small_font,
    )

    d.text(
        (
            600,
            footer_y,
        ),
        "No. of PSS: ________",
        fill=BORDER,
        font=small_font,
    )

    d.text(
        (
            60,
            footer_y + 45,
        ),
        f"Total Capacity (MW): {total}",
        fill=BORDER,
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

    canvas = Image.new(
        "RGB",
        (
            columns
            * thumb_w
            + (columns + 1)
            * gap,

            rows
            * thumb_h
            + (rows + 1)
            * gap,
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
                i % columns
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
                i // columns
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
# IMAGE BYTES
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
# PAGE HEADER
# ============================================================

st.title(
    "☀️ Solar Date Coloring"
)

st.caption(
    "Click a date on the brain, choose Red, Green, Yellow or Blue, and continue to the next date."
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

        if (
            i
            == st.session_state.active_sheet
        ):

            label = f"🟦 {i + 1}"

        else:

            label = str(
                i + 1
            )

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
# MAIN AREA
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
# LEFT CONTROL PANEL
# ============================================================

with controls:

    st.subheader(
        f"Sheet {sheet_index + 1}"
    )

    # --------------------------------------------------------
    # HEADER DETAILS
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
    # DATABASE
    # --------------------------------------------------------

    st.markdown(
        "### Database"
    )

    st.caption(
        "Save sheet details permanently."
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
    # COLOR DATES
    # --------------------------------------------------------

    st.markdown(
        "### Color Dates"
    )

    st.info(
        "Click a date in the preview, then select its color."
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
    # COLORS
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
                    f"color_"
                    f"{sheet_index}_"
                    f"{color_name}"
                ),
                use_container_width=True,
            ):

                if selected_actual_date:

                    # Save immediately
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

                    # ----------------------------------------
                    # Move to next uncolored valid date
                    # ----------------------------------------

                    valid_days = days_in_month(
                        sheet["header_date"]
                    )

                    next_day = None

                    for candidate in range(
                        selected_day + 1,
                        valid_days + 1,
                    ):

                        if (
                            candidate
                            not in sheet["days"]
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
                                not in sheet["days"]
                            ):

                                next_day = candidate
                                break

                    if next_day:

                        st.session_state.selected_day = (
                            next_day
                        )

                    st.session_state.last_click_signature = None

                    st.session_state.map_version += 1

                    st.rerun()

    # --------------------------------------------------------
    # SELECTED DATE STATUS
    # --------------------------------------------------------

    existing = sheet[
        "days"
    ].get(
        selected_day
    )

    if existing:

        st.success(
            f"Date {selected_day}: "
            f"{existing['color_name']}"
        )

    else:

        st.caption(
            f"Date {selected_day} is not colored."
        )

    # --------------------------------------------------------
    # CLEAR SELECTED
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
# RIGHT PREVIEW
# ============================================================

with preview:

    st.subheader(
        "Sheet Preview"
    )

    # --------------------------------------------------------
    # HEADER
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
    # CLICK HANDLING
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

                if not isinstance(
                    point,
                    dict,
                ):
                    continue

                customdata = point.get(
                    "customdata"
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

                if actual_date(
                    sheet,
                    clicked_day,
                ):

                    signature = (
                        sheet_index,
                        clicked_day,
                    )

                    if (
                        signature
                        != st.session_state.last_click_signature
                    ):

                        st.session_state.selected_day = (
                            clicked_day
                        )

                        st.session_state.last_click_signature = (
                            signature
                        )

                        st.session_state.map_version += 1

                        st.rerun()

    # --------------------------------------------------------
    # SELECTED DATE
    # --------------------------------------------------------

    selected_day = (
        st.session_state.selected_day
    )

    selected_date = actual_date(
        sheet,
        selected_day,
    )

    if selected_date:

        existing = sheet[
            "days"
        ].get(
            selected_day
        )

        if existing:

            st.success(
                f"Selected Date: "
                f"{selected_date.strftime('%d-%b-%Y')}  •  "
                f"{existing['color_name']}"
            )

        else:

            st.info(
                f"Selected Date: "
                f"{selected_date.strftime('%d-%b-%Y')}  •  "
                f"Not colored"
            )


# ============================================================
# LEGEND
# ============================================================

st.divider()

st.subheader(
    "Color Legend"
)

legend_cols = st.columns(
    4
)

for col, (
    color_name,
    color_hex,
) in zip(
    legend_cols,
    COLORS.items(),
):

    with col:

        st.markdown(
            f"""
            <div style="
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
                background: {color_hex};
                color: #111111;
                font-weight: 600;
            ">
                {color_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SAVED DATES
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
# FINAL COLLAGE
# ============================================================

st.divider()

st.subheader(
    "Final 10-Sheet Collage"
)

st.caption(
    "All 10 sheets are combined into one 4 + 4 + 2 collage."
)

if st.button(
    "🧠 Generate / Refresh 10-Sheet Collage",
    type="primary",
    use_container_width=True,
):

    st.session_state.show_collage = True


if st.session_state.show_collage:

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
# DATABASE BACKUP CSV
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
