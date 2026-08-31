# ============================================================
# BRAIN PENALTY / SOLAR DATE COLORING
# ============================================================
#
# Features
# ------------------------------------------------------------
# • 10 independent sheets
# • No sidebar
# • Dates, not PSS
# • Header shows Month + Year
# • User writes the left header manually
# • User can enter MAL and Total Capacity
# • Click a date directly on the brain
# • Choose Red / Green / Yellow / Blue
# • Selected color is saved immediately
# • Automatically moves to next uncolored date
# • SQLite database
# • "Save Details to Database" button
# • Database is reloaded when app starts
# • Refined brain-like visual
# • Colored cells touch each other
# • Printable sheet
# • 10-sheet collage
# • CSV backup
#
# IMPORTANT
# ------------------------------------------------------------
# Replace the WHOLE existing page with this file.
# Do not paste this below the previous code.
# ============================================================


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
# CONSTANTS
# ============================================================

N_SHEETS = 10
N_DAYS = 31

COLORS = {
    "Red": "#F26B4F",
    "Green": "#35C98B",
    "Yellow": "#FFF36A",
    "Blue": "#2F80ED",
}

DEFAULT_COLOR = "#F26B4F"

BLACK = "#111111"
GREY = "#777777"
LIGHT_GREY = "#E8E8E8"
BRAIN_BACKGROUND = "#FFFDF7"


# ============================================================
# DATABASE PATH
# ============================================================
#
# SQLite is used so the page does not depend on Streamlit
# session_state for persistence.
#
# If your deployment filesystem is persistent, this survives
# reloads/restarts.
#
# For Streamlit Cloud, local SQLite can be lost when the app
# container is recreated. For true permanent cloud persistence,
# use an external database such as Supabase/PostgreSQL.
# ============================================================

DB_PATH = Path("solar_date_coloring.db")


# ============================================================
# HIDE SIDEBAR
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stSidebar"] {
        display: none !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 3rem;
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
            name TEXT NOT NULL DEFAULT '',
            header_date TEXT NOT NULL DEFAULT '',
            mal TEXT NOT NULL DEFAULT '',
            total TEXT NOT NULL DEFAULT ''
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

    for sheet_no in range(1, N_SHEETS + 1):

        conn.execute(
            """
            INSERT OR IGNORE INTO sheets (
                sheet_no,
                name,
                header_date,
                mal,
                total
            )
            VALUES (?, '', ?, '', '')
            """,
            (
                sheet_no,
                date.today()
                .replace(day=1)
                .isoformat(),
            ),
        )

    conn.commit()

    return conn


DB = get_database()


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def database_load_sheet(sheet_no):

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

        result = {
            "name": "",
            "header_date": date.today().replace(
                day=1
            ),
            "mal": "",
            "total": "",
            "days": {},
        }

    else:

        name = str(row[0] or "")
        raw_header = str(row[1] or "")
        mal = str(row[2] or "")
        total = str(row[3] or "")

        try:

            header_date = date.fromisoformat(
                raw_header
            ).replace(day=1)

        except Exception:

            header_date = date.today().replace(
                day=1
            )

        result = {
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

        try:
            day = int(day)
        except Exception:
            continue

        if (
            1 <= day <= N_DAYS
            and color in COLORS
        ):

            result["days"][day] = {
                "color_name": color,
                "color_hex": COLORS[color],
            }

    return result


def database_save_details(
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
            str(sheet.get("name", "")),
            sheet["header_date"].isoformat(),
            str(sheet.get("mal", "")),
            str(sheet.get("total", "")),
            sheet_no,
        ),
    )

    DB.commit()


def database_save_color(
    sheet_no,
    day,
    color_name,
):

    if color_name not in COLORS:
        return

    if not 1 <= int(day) <= N_DAYS:
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
            int(sheet_no),
            int(day),
            color_name,
        ),
    )

    DB.commit()


def database_delete_color(
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
            int(sheet_no),
            int(day),
        ),
    )

    DB.commit()


def database_clear_sheet(
    sheet_no,
):

    DB.execute(
        """
        DELETE FROM date_colors
        WHERE sheet_no = ?
        """,
        (int(sheet_no),),
    )

    DB.commit()


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

if "database_loaded" not in st.session_state:
    st.session_state.database_loaded = False


# ============================================================
# LOAD ALL SHEETS ONCE
# ============================================================

if not st.session_state.database_loaded:

    st.session_state.sheets = [
        database_load_sheet(i)
        for i in range(
            1,
            N_SHEETS + 1,
        )
    ]

    st.session_state.database_loaded = True


# ============================================================
# SAFE SHEET NORMALIZATION
# ============================================================

def normalize_sheet(sheet):

    if not isinstance(sheet, dict):

        sheet = {}

    name = str(
        sheet.get("name", "") or ""
    )

    mal = str(
        sheet.get("mal", "") or ""
    )

    total = str(
        sheet.get("total", "") or ""
    )

    raw_date = sheet.get(
        "header_date",
        date.today().replace(day=1),
    )

    if isinstance(raw_date, date):

        header_date = raw_date.replace(
            day=1
        )

    else:

        try:

            header_date = date.fromisoformat(
                str(raw_date)
            ).replace(day=1)

        except Exception:

            header_date = date.today().replace(
                day=1
            )

    clean_days = {}

    raw_days = sheet.get(
        "days",
        {},
    )

    if isinstance(raw_days, dict):

        for raw_day, raw_value in raw_days.items():

            try:
                day = int(raw_day)
            except Exception:
                continue

            if not 1 <= day <= N_DAYS:
                continue

            if not isinstance(
                raw_value,
                dict,
            ):
                continue

            color_name = str(
                raw_value.get(
                    "color_name",
                    "",
                )
                or ""
            )

            if color_name in COLORS:

                clean_days[day] = {
                    "color_name": color_name,
                    "color_hex": COLORS[color_name],
                }

    return {
        "name": name,
        "header_date": header_date,
        "mal": mal,
        "total": total,
        "days": clean_days,
    }


st.session_state.sheets = [
    normalize_sheet(sheet)
    for sheet in st.session_state.sheets
]


# ============================================================
# BASIC HELPERS
# ============================================================

def current_sheet():

    return st.session_state.sheets[
        st.session_state.active_sheet
    ]


def days_in_current_month(sheet):

    return calendar.monthrange(
        sheet["header_date"].year,
        sheet["header_date"].month,
    )[1]


def get_actual_date(
    sheet,
    day,
):

    try:

        return date(
            sheet["header_date"].year,
            sheet["header_date"].month,
            int(day),
        )

    except Exception:

        return None


def polygon_center(points):

    """
    Calculate the center of a polygon.

    This function is intentionally defined before
    make_map() and render_sheet() so there is no
    NameError.
    """

    if not points:

        return (
            0,
            0,
        )

    x_values = [
        p[0]
        for p in points
    ]

    y_values = [
        p[1]
        for p in points
    ]

    return (
        sum(x_values) / len(x_values),
        sum(y_values) / len(y_values),
    )


# Keep compatibility with any internal reference to center().
center = polygon_center


# ============================================================
# BRAIN OUTLINE
# ============================================================

def get_brain_outline():

    return [

        # Left upper
        (110, 190),
        (125, 155),
        (165, 125),
        (220, 103),
        (285, 92),
        (350, 91),
        (410, 102),
        (460, 126),
        (500, 148),

        # Right upper
        (540, 126),
        (590, 102),
        (650, 91),
        (715, 92),
        (780, 103),
        (835, 125),
        (875, 155),
        (890, 190),

        # Right side
        (908, 245),
        (912, 315),
        (902, 380),
        (914, 450),
        (908, 525),
        (918, 600),
        (910, 680),
        (916, 755),
        (900, 830),
        (878, 900),
        (845, 960),
        (795, 1015),
        (730, 1055),
        (665, 1080),
        (590, 1093),
        (550, 1085),

        # Bottom center
        (500, 1065),

        # Left bottom
        (450, 1085),
        (410, 1093),
        (335, 1080),
        (270, 1055),
        (205, 1015),
        (155, 960),
        (122, 900),
        (100, 830),
        (84, 755),
        (90, 680),
        (82, 600),
        (92, 525),
        (86, 450),
        (98, 380),
        (88, 315),
        (92, 245),
    ]


BRAIN_OUTLINE = get_brain_outline()


# ============================================================
# DATE CELL GEOMETRY
# ============================================================
#
# The date blocks are intentionally adjoining.
# There are no gaps between colored regions.
# ============================================================

def build_date_cells():

    cells = {}

    # --------------------------------------------------------
    # TOP LEFT / RIGHT
    # --------------------------------------------------------

    cells[1] = [
        (295, 150),
        (500, 150),
        (500, 240),
        (280, 240),
    ]

    cells[2] = [
        (500, 150),
        (705, 150),
        (720, 240),
        (500, 240),
    ]

    # --------------------------------------------------------
    # SECOND BAND
    # --------------------------------------------------------

    second_row = [
        3,
        4,
        5,
        6,
    ]

    left = 250
    right = 750
    width = (
        right - left
    ) / 4

    for index, day in enumerate(
        second_row
    ):

        x1 = left + (
            index * width
        )

        x2 = left + (
            (index + 1) * width
        )

        cells[day] = [
            (
                x1,
                240,
            ),
            (
                x2,
                240,
            ),
            (
                x2,
                335,
            ),
            (
                x1,
                335,
            ),
        ]

    # --------------------------------------------------------
    # MAIN BODY
    # --------------------------------------------------------

    main_rows = [
        (
            [7, 8, 9, 10],
            335,
            430,
        ),
        (
            [11, 12, 13, 14],
            430,
            525,
        ),
        (
            [15, 16, 17, 18],
            525,
            620,
        ),
        (
            [19, 20, 21, 22],
            620,
            715,
        ),
        (
            [23, 24, 25, 26],
            715,
            810,
        ),
        (
            [27, 28, 29, 30],
            810,
            905,
        ),
    ]

    left = 145
    right = 855

    for days, top, bottom in main_rows:

        width = (
            right - left
        ) / len(days)

        for index, day in enumerate(days):

            x1 = (
                left
                + index * width
            )

            x2 = (
                left
                + (index + 1) * width
            )

            cells[day] = [
                (
                    x1,
                    top,
                ),
                (
                    x2,
                    top,
                ),
                (
                    x2,
                    bottom,
                ),
                (
                    x1,
                    bottom,
                ),
            ]

    # --------------------------------------------------------
    # BOTTOM DATE
    # --------------------------------------------------------

    cells[31] = [
        (340, 905),
        (660, 905),
        (650, 1000),
        (350, 1000),
    ]

    return cells


DATE_CELLS = build_date_cells()


# ============================================================
# INTERNAL BRAIN FOLDS
# ============================================================

def get_brain_folds():

    return [

        # LEFT
        [
            (100, 240),
            (155, 205),
            (220, 225),
            (280, 180),
        ],

        [
            (92, 330),
            (150, 290),
            (215, 315),
            (270, 265),
        ],

        [
            (88, 420),
            (150, 380),
            (215, 405),
            (270, 355),
        ],

        [
            (85, 515),
            (150, 475),
            (220, 500),
            (270, 450),
        ],

        [
            (82, 620),
            (150, 575),
            (220, 610),
            (275, 555),
        ],

        [
            (85, 720),
            (150, 680),
            (220, 710),
            (275, 665),
        ],

        [
            (100, 820),
            (160, 780),
            (220, 815),
            (285, 765),
        ],

        [
            (130, 910),
            (185, 875),
            (245, 905),
            (310, 860),
        ],

        # RIGHT
        [
            (900, 240),
            (845, 205),
            (780, 225),
            (720, 180),
        ],

        [
            (908, 330),
            (850, 290),
            (785, 315),
            (730, 265),
        ],

        [
            (912, 420),
            (850, 380),
            (785, 405),
            (730, 355),
        ],

        [
            (915, 515),
            (850, 475),
            (780, 500),
            (730, 450),
        ],

        [
            (918, 620),
            (850, 575),
            (780, 610),
            (725, 555),
        ],

        [
            (915, 720),
            (850, 680),
            (780, 710),
            (725, 665),
        ],

        [
            (900, 820),
            (840, 780),
            (780, 815),
            (715, 765),
        ],

        [
            (870, 910),
            (815, 875),
            (755, 905),
            (690, 860),
        ],
    ]


BRAIN_FOLDS = get_brain_folds()


# ============================================================
# MAP / PLOTLY PREVIEW
# ============================================================

def make_map(sheet):

    fig = go.Figure()

    valid_days = days_in_current_month(
        sheet
    )

    # --------------------------------------------------------
    # Brain background
    # --------------------------------------------------------

    outline_x = [
        point[0]
        for point in BRAIN_OUTLINE
    ]

    outline_y = [
        point[1]
        for point in BRAIN_OUTLINE
    ]

    fig.add_trace(
        go.Scatter(
            x=outline_x + [
                outline_x[0]
            ],
            y=outline_y + [
                outline_y[0]
            ],
            mode="lines",
            fill="toself",
            fillcolor=BRAIN_BACKGROUND,
            line=dict(
                color=BLACK,
                width=5,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Date blocks
    # --------------------------------------------------------

    for day in range(
        1,
        N_DAYS + 1,
    ):

        if day not in DATE_CELLS:
            continue

        points = DATE_CELLS[day]

        is_valid = (
            day <= valid_days
        )

        assignment = sheet[
            "days"
        ].get(day)

        if assignment:

            fill = assignment[
                "color_hex"
            ]

        elif is_valid:

            fill = DEFAULT_COLOR

        else:

            fill = "#E2E2E2"

        xs = [
            point[0]
            for point in points
        ]

        ys = [
            point[1]
            for point in points
        ]

        xs.append(
            xs[0]
        )

        ys.append(
            ys[0]
        )

        actual = get_actual_date(
            sheet,
            day,
        )

        if actual:

            date_label = actual.strftime(
                "%d-%b-%Y"
            )

        else:

            date_label = "Invalid date"

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
                    f"<br>{date_label}"
                    "<br><br>"
                    "Click to select"
                    "<extra></extra>"
                ),
                showlegend=False,
                opacity=(
                    1
                    if is_valid
                    else 0.30
                ),
            )
        )

        # ----------------------------------------------------
        # Date number
        # ----------------------------------------------------

        cx, cy = polygon_center(
            points
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
                    size=40,
                    color=(
                        "rgba(255,255,255,0.01)"
                    ),
                    line=dict(
                        width=0
                    ),
                ),
                textfont=dict(
                    family="Arial",
                    size=17,
                    color=BLACK,
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
    # Horizontal shared boundaries
    # --------------------------------------------------------

    boundaries = [
        (
            240,
            280,
            720,
        ),
        (
            335,
            145,
            855,
        ),
        (
            430,
            145,
            855,
        ),
        (
            525,
            145,
            855,
        ),
        (
            620,
            145,
            855,
        ),
        (
            715,
            145,
            855,
        ),
        (
            810,
            145,
            855,
        ),
        (
            905,
            145,
            855,
        ),
    ]

    for y, x1, x2 in boundaries:

        fig.add_trace(
            go.Scatter(
                x=[
                    x1,
                    x2,
                ],
                y=[
                    y,
                    y,
                ],
                mode="lines",
                line=dict(
                    color=BLACK,
                    width=2,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # Vertical center
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                500,
                500,
            ],
            y=[
                150,
                1065,
            ],
            mode="lines",
            line=dict(
                color=BLACK,
                width=4,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Brain folds
    # --------------------------------------------------------

    for fold in BRAIN_FOLDS:

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
                    color=GREY,
                    width=2,
                    shape="spline",
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
            x=outline_x + [
                outline_x[0]
            ],
            y=outline_y + [
                outline_y[0]
            ],
            mode="lines",
            line=dict(
                color=BLACK,
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
        fixedrange=True,
        range=[
            45,
            955,
        ],
    )

    fig.update_yaxes(
        visible=False,
        fixedrange=True,
        range=[
            1140,
            60,
        ],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        height=780,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        clickmode="event+select",
        dragmode=False,
        showlegend=False,
    )

    return fig


# ============================================================
# FONT
# ============================================================

def get_font(
    size,
    bold=False,
):

    if bold:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]

    else:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for path in candidates:

        try:

            return ImageFont.truetype(
                path,
                size,
            )

        except Exception:
            pass

    return ImageFont.load_default()


# ============================================================
# PRINTABLE BRAIN SHEET
# ============================================================

def render_sheet(sheet):

    width = 1000
    height = 1300

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        image
    )

    title_font = get_font(
        34,
        True,
    )

    month_font = get_font(
        30,
        True,
    )

    day_font = get_font(
        21,
        True,
    )

    footer_font = get_font(
        19,
        False,
    )

    # --------------------------------------------------------
    # Page border
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            15,
            15,
            width - 15,
            height - 15,
        ),
        radius=8,
        outline=BLACK,
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

    draw.text(
        (
            60,
            48,
        ),
        title,
        fill=BLACK,
        font=title_font,
    )

    draw.line(
        (
            55,
            96,
            350,
            96,
        ),
        fill=BLACK,
        width=2,
    )

    month_box = draw.textbbox(
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

    draw.text(
        (
            width
            - 60
            - month_width,
            50,
        ),
        month,
        fill=BLACK,
        font=month_font,
    )

    # --------------------------------------------------------
    # Brain base
    # --------------------------------------------------------

    draw.polygon(
        BRAIN_OUTLINE,
        fill=BRAIN_BACKGROUND,
    )

    valid_days = days_in_current_month(
        sheet
    )

    # --------------------------------------------------------
    # Date fills
    # --------------------------------------------------------

    for day, points in DATE_CELLS.items():

        assignment = sheet[
            "days"
        ].get(day)

        if assignment:

            fill = assignment[
                "color_hex"
            ]

        elif day <= valid_days:

            fill = DEFAULT_COLOR

        else:

            fill = "#E2E2E2"

        draw.polygon(
            points,
            fill=fill,
        )

    # --------------------------------------------------------
    # Shared borders
    # --------------------------------------------------------

    boundaries = [
        (
            240,
            280,
            720,
        ),
        (
            335,
            145,
            855,
        ),
        (
            430,
            145,
            855,
        ),
        (
            525,
            145,
            855,
        ),
        (
            620,
            145,
            855,
        ),
        (
            715,
            145,
            855,
        ),
        (
            810,
            145,
            855,
        ),
        (
            905,
            145,
            855,
        ),
    ]

    for y, x1, x2 in boundaries:

        draw.line(
            (
                x1,
                y,
                x2,
                y,
            ),
            fill=BLACK,
            width=2,
        )

    # --------------------------------------------------------
    # Brain folds
    # --------------------------------------------------------

    for fold in BRAIN_FOLDS:

        draw.line(
            fold,
            fill=GREY,
            width=2,
            joint="curve",
        )

    # --------------------------------------------------------
    # Center line
    # --------------------------------------------------------

    draw.line(
        (
            500,
            150,
            500,
            1065,
        ),
        fill=BLACK,
        width=4,
    )

    # --------------------------------------------------------
    # Date numbers
    # --------------------------------------------------------

    for day, points in DATE_CELLS.items():

        cx, cy = polygon_center(
            points
        )

        text = str(day)

        bbox = draw.textbbox(
            (
                0,
                0,
            ),
            text,
            font=day_font,
        )

        text_width = (
            bbox[2]
            - bbox[0]
        )

        text_height = (
            bbox[3]
            - bbox[1]
        )

        draw.text(
            (
                cx
                - text_width / 2,
                cy
                - text_height / 2,
            ),
            text,
            fill=BLACK,
            font=day_font,
        )

    # --------------------------------------------------------
    # Outer brain border
    # --------------------------------------------------------

    draw.line(
        BRAIN_OUTLINE
        + [
            BRAIN_OUTLINE[0]
        ],
        fill=BLACK,
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

    draw.text(
        (
            60,
            footer_y,
        ),
        f"MAL Capacity (MW): {mal}",
        fill=BLACK,
        font=footer_font,
    )

    draw.text(
        (
            600,
            footer_y,
        ),
        "No. of PSS: ________",
        fill=BLACK,
        font=footer_font,
    )

    draw.text(
        (
            60,
            footer_y + 45,
        ),
        f"Total Capacity (MW): {total}",
        fill=BLACK,
        font=footer_font,
    )

    return image


# ============================================================
# COLLAGE
# ============================================================

def create_collage(
    images,
    columns=4,
):

    gap = 20

    thumb_width = 520
    thumb_height = 676

    rows = int(
        np.ceil(
            len(images)
            / columns
        )
    )

    collage_width = (
        columns
        * thumb_width
        + (
            columns + 1
        )
        * gap
    )

    collage_height = (
        rows
        * thumb_height
        + (
            rows + 1
        )
        * gap
    )

    collage_image = Image.new(
        "RGB",
        (
            collage_width,
            collage_height,
        ),
        "#E7E7E7",
    )

    for index, image in enumerate(
        images
    ):

        copy = image.copy()

        copy.thumbnail(
            (
                thumb_width,
                thumb_height,
            ),
            Image.Resampling.LANCZOS,
        )

        x = (
            gap
            + (
                index
                % columns
            )
            * (
                thumb_width
                + gap
            )
            + (
                thumb_width
                - copy.width
            )
            // 2
        )

        y = (
            gap
            + (
                index
                // columns
            )
            * (
                thumb_height
                + gap
            )
            + (
                thumb_height
                - copy.height
            )
            // 2
        )

        collage_image.paste(
            copy,
            (
                x,
                y,
            ),
        )

    return collage_image


# ============================================================
# PNG BYTES
# ============================================================

def to_png_bytes(
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
# MAIN TITLE
# ============================================================

st.title(
    "☀️ Solar Date Coloring"
)

st.caption(
    "Click a date on the brain, choose a color, and continue to the next date."
)


# ============================================================
# SHEET NAVIGATION
# ============================================================

st.subheader(
    "Sheets"
)

navigation_columns = st.columns(
    N_SHEETS
)

for index, column in enumerate(
    navigation_columns
):

    with column:

        if (
            index
            == st.session_state.active_sheet
        ):

            button_text = (
                f"● Sheet {index + 1}"
            )

        else:

            button_text = (
                f"Sheet {index + 1}"
            )

        if st.button(
            button_text,
            key=f"sheet_button_{index}",
            use_container_width=True,
        ):

            st.session_state.active_sheet = (
                index
            )

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
# MAIN CONTENT
# ============================================================

st.divider()

left_panel, right_panel = st.columns(
    [
        0.75,
        1.65,
    ],
    gap="large",
)


# ============================================================
# LEFT PANEL
# ============================================================

with left_panel:

    st.subheader(
        f"Sheet {sheet_index + 1}"
    )

    # --------------------------------------------------------
    # Header input
    # --------------------------------------------------------

    new_name = st.text_input(
        "Write on left header",
        value=sheet["name"],
        placeholder="Solar - G1",
        key=f"name_{sheet_index}",
    )

    sheet["name"] = new_name

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    new_month = st.date_input(
        "Header month",
        value=sheet["header_date"],
        key=f"month_{sheet_index}",
    )

    sheet["header_date"] = new_month.replace(
        day=1
    )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    sheet["mal"] = st.text_input(
        "MAL Capacity (MW)",
        value=sheet["mal"],
        key=f"mal_{sheet_index}",
    )

    sheet["total"] = st.text_input(
        "Total Capacity (MW)",
        value=sheet["total"],
        key=f"total_{sheet_index}",
    )

    # --------------------------------------------------------
    # Database button
    # --------------------------------------------------------

    st.markdown(
        "### Database"
    )

    if st.button(
        "💾 Save Details to Database",
        type="primary",
        use_container_width=True,
        key=f"save_{sheet_index}",
    ):

        database_save_details(
            sheet_index + 1,
            sheet,
        )

        # Reload from database
        st.session_state.sheets[
            sheet_index
        ] = database_load_sheet(
            sheet_index + 1
        )

        st.success(
            "Sheet details saved."
        )

        st.rerun()

    # --------------------------------------------------------
    # Date coloring
    # --------------------------------------------------------

    st.markdown(
        "### Color Dates"
    )

    st.caption(
        "1. Click a date in the brain"
        "\n\n"
        "2. Choose Red, Green, Yellow or Blue"
    )

    selected_day = (
        st.session_state.selected_day
    )

    selected_date = get_actual_date(
        sheet,
        selected_day,
    )

    if selected_date:

        st.write(
            f"**Selected Date:** "
            f"{selected_date.strftime('%d-%b-%Y')}"
        )

    else:

        st.warning(
            f"Day {selected_day} does not exist in this month."
        )

    # --------------------------------------------------------
    # Color buttons
    # --------------------------------------------------------

    color_columns = st.columns(
        4
    )

    for color_name, color_column in zip(
        COLORS.keys(),
        color_columns,
    ):

        with color_column:

            if st.button(
                color_name,
                key=(
                    f"color_"
                    f"{sheet_index}_"
                    f"{color_name}"
                ),
                use_container_width=True,
            ):

                selected_date = get_actual_date(
                    sheet,
                    selected_day,
                )

                if selected_date is not None:

                    # ----------------------------------------
                    # Save immediately
                    # ----------------------------------------

                    database_save_color(
                        sheet_index + 1,
                        selected_day,
                        color_name,
                    )

                    sheet[
                        "days"
                    ][selected_day] = {
                        "color_name": color_name,
                        "color_hex": COLORS[
                            color_name
                        ],
                    }

                    # ----------------------------------------
                    # Automatically find next date
                    # ----------------------------------------

                    valid_days = days_in_current_month(
                        sheet
                    )

                    next_day = None

                    # Forward search
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

                    # Start from beginning
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

                    if next_day is not None:

                        st.session_state.selected_day = (
                            next_day
                        )

                    st.session_state.last_click_signature = None

                    st.session_state.map_version += 1

                    st.rerun()

    # --------------------------------------------------------
    # Current date status
    # --------------------------------------------------------

    current_assignment = sheet[
        "days"
    ].get(
        selected_day
    )

    if current_assignment:

        st.success(
            f"Date {selected_day} is "
            f"{current_assignment['color_name']}"
        )

    else:

        st.info(
            f"Date {selected_day} is not colored."
        )

    # --------------------------------------------------------
    # Clear selected
    # --------------------------------------------------------

    if st.button(
        "🗑 Clear Selected Date",
        use_container_width=True,
        key=f"clear_date_{sheet_index}",
    ):

        database_delete_color(
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
    # Clear complete sheet
    # --------------------------------------------------------

    if st.button(
        "🗑 Clear All Dates",
        use_container_width=True,
        key=f"clear_all_{sheet_index}",
    ):

        database_clear_sheet(
            sheet_index + 1
        )

        sheet[
            "days"
        ] = {}

        st.session_state.selected_day = 1

        st.session_state.map_version += 1

        st.rerun()


# ============================================================
# RIGHT PANEL
# ============================================================

with right_panel:

    st.subheader(
        "Brain Preview"
    )

    # --------------------------------------------------------
    # Preview header
    # --------------------------------------------------------

    header_left, header_right = st.columns(
        [
            3,
            1,
        ]
    )

    with header_left:

        st.markdown(
            f"### {sheet['name'] or 'Solar - ____'}"
        )

    with header_right:

        st.markdown(
            f"### {sheet['header_date'].strftime('%B %Y')}"
        )

    # --------------------------------------------------------
    # Brain map
    # --------------------------------------------------------

    figure = make_map(
        sheet
    )

    event = st.plotly_chart(
        figure,
        use_container_width=True,
        key=(
            f"brain_"
            f"{sheet_index}_"
            f"{st.session_state.map_version}"
        ),
        on_select="rerun",
        selection_mode=[
            "points"
        ],
    )

    # --------------------------------------------------------
    # Detect clicked date
    # --------------------------------------------------------

    if event is not None:

        try:

            points = (
                event.selection.points
            )

        except Exception:

            points = []

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

                except Exception:

                    continue

            if (
                clicked_day is not None
                and 1 <= clicked_day <= N_DAYS
            ):

                if get_actual_date(
                    sheet,
                    clicked_day,
                ) is not None:

                    click_signature = (
                        sheet_index,
                        clicked_day,
                    )

                    if (
                        click_signature
                        != st.session_state.last_click_signature
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
    # Selected date indicator
    # --------------------------------------------------------

    selected_day = (
        st.session_state.selected_day
    )

    selected_date = get_actual_date(
        sheet,
        selected_day,
    )

    if selected_date:

        assignment = sheet[
            "days"
        ].get(
            selected_day
        )

        if assignment:

            st.success(
                f"Selected: "
                f"{selected_date.strftime('%d-%b-%Y')}  •  "
                f"{assignment['color_name']}"
            )

        else:

            st.info(
                f"Selected: "
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

legend_columns = st.columns(
    4
)

for column, (
    color_name,
    color_hex,
) in zip(
    legend_columns,
    COLORS.items(),
):

    with column:

        st.markdown(
            f"""
            <div style="
                background:{color_hex};
                border:2px solid #111;
                border-radius:8px;
                padding:10px;
                text-align:center;
                font-weight:600;
                color:#111;
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
    sheet["days"].keys()
):

    actual = get_actual_date(
        sheet,
        day,
    )

    if actual is None:
        continue

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

    saved_df = pd.DataFrame(
        saved_rows
    )

    st.dataframe(
        saved_df,
        hide_index=True,
        use_container_width=True,
    )

else:

    st.caption(
        "No dates colored yet."
    )


# ============================================================
# 10 SHEET COLLAGE
# ============================================================

st.divider()

st.subheader(
    "Final 10-Sheet Collage"
)

st.caption(
    "The final output contains all 10 sheets in a 4 + 4 + 2 layout."
)

if st.button(
    "🧠 Generate / Refresh 10-Sheet Collage",
    type="primary",
    use_container_width=True,
):

    st.session_state.show_collage = True


if st.session_state.show_collage:

    # Always reload from database.
    # This prevents the collage from depending
    # only on session state.

    database_sheets = [
        database_load_sheet(i)
        for i in range(
            1,
            N_SHEETS + 1,
        )
    ]

    rendered_sheets = [
        render_sheet(
            sheet_data
        )
        for sheet_data in database_sheets
    ]

    final_collage = create_collage(
        rendered_sheets,
        columns=4,
    )

    st.image(
        final_collage,
        use_container_width=True,
    )

    st.download_button(
        "📥 Download 10-Sheet Collage",
        data=to_png_bytes(
            final_collage
        ),
        file_name=(
            "solar_date_10_sheet_collage.png"
        ),
        mime="image/png",
        use_container_width=True,
    )


# ============================================================
# DATABASE BACKUP
# ============================================================

st.divider()

st.subheader(
    "Database Backup"
)

backup_rows = []

for sheet_no in range(
    1,
    N_SHEETS + 1,
):

    db_sheet = database_load_sheet(
        sheet_no
    )

    for day in range(
        1,
        N_DAYS + 1,
    ):

        actual = get_actual_date(
            db_sheet,
            day,
        )

        assignment = db_sheet[
            "days"
        ].get(
            day
        )

        backup_rows.append(
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
    backup_rows
)

st.download_button(
    "📥 Download Database Backup CSV",
    data=backup_df.to_csv(
        index=False
    ),
    file_name=(
        "solar_date_coloring_backup.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)
