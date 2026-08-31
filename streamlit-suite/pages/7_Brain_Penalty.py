# ============================================================
# 7_Brain_Penalty.py
# SOLAR DATE COLORING
# ============================================================

import io
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Date Coloring",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_SHEETS = 10

# IMPORTANT:
# For Streamlit Cloud, change this to a persistent database later.
DB_PATH = Path("solar_brain.db")

COLORS = {
    "Red": "#E53935",
    "Green": "#43A047",
    "Yellow": "#FDD835",
    "Blue": "#1E88E5",
}

EMOJIS = {
    "Red": "🔴",
    "Green": "🟢",
    "Yellow": "🟡",
    "Blue": "🔵",
}


# ============================================================
# DATABASE
# ============================================================

def get_db():
    return sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False
    )


def initialize_database():

    conn = get_db()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # SHEETS TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sheets (
            sheet_no INTEGER PRIMARY KEY,
            header_text TEXT DEFAULT '',
            month_year TEXT DEFAULT '',
            mal_capacity TEXT DEFAULT '',
            total_capacity TEXT DEFAULT ''
        )
        """
    )

    # --------------------------------------------------------
    # DATE COLOR TABLE
    # --------------------------------------------------------

    cursor.execute(
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

    # --------------------------------------------------------
    # CREATE 10 SHEETS
    # --------------------------------------------------------

    for sheet_no in range(
        1,
        TOTAL_SHEETS + 1
    ):

        cursor.execute(
            """
            INSERT OR IGNORE INTO sheets (
                sheet_no
            )
            VALUES (?)
            """,
            (sheet_no,)
        )

    conn.commit()
    conn.close()


initialize_database()


# ============================================================
# LOAD SHEET
# ============================================================

def load_sheet(sheet_no):

    conn = get_db()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # SHEET DETAILS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            header_text,
            month_year,
            mal_capacity,
            total_capacity

        FROM sheets

        WHERE sheet_no = ?
        """,
        (sheet_no,)
    )

    row = cursor.fetchone()

    if row:

        header_text = row[0] or ""
        month_year = row[1] or ""
        mal_capacity = row[2] or ""
        total_capacity = row[3] or ""

    else:

        header_text = ""
        month_year = ""
        mal_capacity = ""
        total_capacity = ""

    # --------------------------------------------------------
    # DATE COLORS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            day,
            color

        FROM date_colors

        WHERE sheet_no = ?

        ORDER BY day
        """,
        (sheet_no,)
    )

    date_rows = cursor.fetchall()

    date_colors = {}

    for day, color in date_rows:

        date_colors[int(day)] = color

    conn.close()

    return {
        "header_text": header_text,
        "month_year": month_year,
        "mal_capacity": mal_capacity,
        "total_capacity": total_capacity,
        "date_colors": date_colors,
    }


# ============================================================
# LOAD ALL SHEETS
# ============================================================

def load_all_sheets():

    sheets = []

    for sheet_no in range(
        1,
        TOTAL_SHEETS + 1
    ):

        sheets.append(
            load_sheet(sheet_no)
        )

    return sheets


# ============================================================
# SAVE SHEET DETAILS
# ============================================================

def save_sheet_details(
    sheet_no,
    header_text,
    month_year,
    mal_capacity,
    total_capacity,
):

    conn = get_db()

    conn.execute(
        """
        UPDATE sheets

        SET
            header_text = ?,
            month_year = ?,
            mal_capacity = ?,
            total_capacity = ?

        WHERE sheet_no = ?
        """,
        (
            header_text,
            month_year,
            mal_capacity,
            total_capacity,
            sheet_no,
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# SAVE DATE COLOR
# ============================================================

def save_date_color(
    sheet_no,
    day,
    color,
):

    if color not in COLORS:
        return

    conn = get_db()

    conn.execute(
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
            color,
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# DELETE DATE COLOR
# ============================================================

def delete_date_color(
    sheet_no,
    day,
):

    conn = get_db()

    conn.execute(
        """
        DELETE FROM date_colors

        WHERE
            sheet_no = ?
            AND day = ?
        """,
        (
            sheet_no,
            day,
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# CLEAR SHEET
# ============================================================

def clear_sheet_dates(sheet_no):

    conn = get_db()

    conn.execute(
        """
        DELETE FROM date_colors

        WHERE sheet_no = ?
        """,
        (sheet_no,)
    )

    conn.commit()
    conn.close()


# ============================================================
# MONTH/YEAR
# ============================================================

def create_month_list():

    values = []

    for year in range(
        2024,
        2036
    ):

        for month in range(
            1,
            13
        ):

            values.append(
                datetime(
                    year,
                    month,
                    1
                ).strftime(
                    "%b-%Y"
                )
            )

    return values


MONTH_LIST = create_month_list()


# ============================================================
# DAYS IN MONTH
# ============================================================

def days_in_month(month_year):

    if not month_year:
        return 31

    try:

        selected = datetime.strptime(
            month_year,
            "%b-%Y"
        )

        if selected.month == 12:

            next_month = datetime(
                selected.year + 1,
                1,
                1
            )

        else:

            next_month = datetime(
                selected.year,
                selected.month + 1,
                1
            )

        current_month = datetime(
            selected.year,
            selected.month,
            1
        )

        return (
            next_month - current_month
        ).days

    except Exception:

        return 31


# ============================================================
# FONT
# ============================================================

def get_font(
    size,
    bold=False
):

    if bold:

        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]

    else:

        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for path in paths:

        if Path(path).exists():

            return ImageFont.truetype(
                path,
                size
            )

    return ImageFont.load_default()


# ============================================================
# DRAW DATE CELL
# ============================================================

def draw_date_cell(
    draw,
    day,
    center_x,
    center_y,
    color,
):

    radius_x = 43
    radius_y = 31

    if color in COLORS:

        fill = COLORS[color]

    else:

        fill = "white"

    draw.rounded_rectangle(
        (
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        ),
        radius=10,
        fill=fill,
        outline="#222222",
        width=3,
    )

    text = str(day)

    font = get_font(
        25,
        True
    )

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_width = (
        bbox[2] - bbox[0]
    )

    text_height = (
        bbox[3] - bbox[1]
    )

    draw.text(
        (
            center_x - text_width / 2,
            center_y - text_height / 2 - 2,
        ),
        text,
        fill="#111111",
        font=font,
    )


# ============================================================
# CREATE BRAIN SHEET IMAGE
# ============================================================

def create_sheet_image(
    sheet_no,
    sheet_data,
):

    WIDTH = 1400
    HEIGHT = 1000

    image = Image.new(
        "RGB",
        (
            WIDTH,
            HEIGHT,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        image
    )

    # --------------------------------------------------------
    # FONTS
    # --------------------------------------------------------

    header_font = get_font(
        42,
        True
    )

    month_font = get_font(
        34,
        True
    )

    date_font = get_font(
        24,
        True
    )

    footer_font = get_font(
        23,
        True
    )

    footer_value_font = get_font(
        22,
        False
    )

    # --------------------------------------------------------
    # OUTER PAPER
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            20,
            20,
            WIDTH - 20,
            HEIGHT - 20,
        ),
        radius=18,
        fill="white",
        outline="#222222",
        width=4,
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    header_text = (
        sheet_data["header_text"]
        or f"Solar - {sheet_no}"
    )

    month_year = (
        sheet_data["month_year"]
        or "Month-Year"
    )

    draw.text(
        (
            65,
            52,
        ),
        header_text,
        fill="#111111",
        font=header_font,
    )

    month_bbox = draw.textbbox(
        (0, 0),
        month_year,
        font=month_font,
    )

    month_width = (
        month_bbox[2]
        - month_bbox[0]
    )

    draw.text(
        (
            WIDTH - month_width - 65,
            58,
        ),
        month_year,
        fill="#111111",
        font=month_font,
    )

    draw.line(
        (
            55,
            125,
            WIDTH - 55,
            125,
        ),
        fill="#222222",
        width=3,
    )

    # --------------------------------------------------------
    # BRAIN OUTLINE
    # --------------------------------------------------------

    left_lobe = (
        70,
        170,
        700,
        795,
    )

    right_lobe = (
        700,
        170,
        1330,
        795,
    )

    draw.ellipse(
        left_lobe,
        fill="white",
        outline="#333333",
        width=5,
    )

    draw.ellipse(
        right_lobe,
        fill="white",
        outline="#333333",
        width=5,
    )

    # --------------------------------------------------------
    # CENTRAL GROOVE
    # --------------------------------------------------------

    draw.line(
        (
            WIDTH // 2,
            195,
            WIDTH // 2,
            760,
        ),
        fill="white",
        width=18,
    )

    draw.line(
        (
            WIDTH // 2,
            195,
            WIDTH // 2,
            760,
        ),
        fill="#333333",
        width=4,
    )

    # --------------------------------------------------------
    # BRAIN FOLDS
    # --------------------------------------------------------

    fold_color = "#999999"

    left_folds = [
        (115, 240, 350, 305),
        (365, 220, 610, 285),
        (100, 365, 340, 430),
        (375, 350, 620, 420),
        (115, 500, 345, 570),
        (380, 490, 620, 565),
        (190, 645, 460, 710),
    ]

    right_folds = [
        (790, 240, 1025, 305),
        (1050, 220, 1295, 285),
        (780, 365, 1020, 430),
        (1050, 350, 1300, 420),
        (780, 500, 1015, 570),
        (1050, 490, 1295, 565),
        (940, 645, 1210, 710),
    ]

    for box in left_folds:

        draw.arc(
            box,
            180,
            355,
            fill=fold_color,
            width=3,
        )

    for box in right_folds:

        draw.arc(
            box,
            185,
            360,
            fill=fold_color,
            width=3,
        )

    # --------------------------------------------------------
    # DATE POSITIONS
    # --------------------------------------------------------

    positions = []

    # LEFT BRAIN
    left_rows = [
        (5, 145, 265),
        (5, 145, 375),
        (5, 145, 485),
        (5, 145, 595),
        (4, 195, 700),
    ]

    day = 1

    for count, start_x, y in left_rows:

        for j in range(count):

            positions.append(
                (
                    day,
                    start_x + j * 105,
                    y,
                )
            )

            day += 1

    # RIGHT BRAIN
    right_rows = [
        (5, 775, 265),
        (5, 775, 375),
        (5, 775, 485),
        (5, 775, 595),
        (4, 825, 700),
    ]

    for count, start_x, y in right_rows:

        for j in range(count):

            positions.append(
                (
                    day,
                    start_x + j * 105,
                    y,
                )
            )

            day += 1

    # --------------------------------------------------------
    # DRAW DATES
    # --------------------------------------------------------

    number_of_days = days_in_month(
        sheet_data["month_year"]
    )

    for day, x, y in positions:

        if day > number_of_days:

            continue

        color = sheet_data[
            "date_colors"
        ].get(day)

        draw_date_cell(
            draw,
            day,
            x,
            y,
            color,
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    footer_y = 830

    draw.line(
        (
            55,
            footer_y,
            WIDTH - 55,
            footer_y,
        ),
        fill="#222222",
        width=3,
    )

    # MAL
    draw.text(
        (
            70,
            footer_y + 28,
        ),
        "MAL Capacity (MW)",
        fill="#111111",
        font=footer_font,
    )

    draw.line(
        (
            70,
            footer_y + 70,
            420,
            footer_y + 70,
        ),
        fill="#333333",
        width=2,
    )

    if sheet_data[
        "mal_capacity"
    ]:

        draw.text(
            (
                70,
                footer_y + 78,
            ),
            sheet_data[
                "mal_capacity"
            ],
            fill="#111111",
            font=footer_value_font,
        )

    # NUMBER OF DATES
    draw.text(
        (
            535,
            footer_y + 28,
        ),
        "No. of Dates",
        fill="#111111",
        font=footer_font,
    )

    draw.line(
        (
            535,
            footer_y + 70,
            850,
            footer_y + 70,
        ),
        fill="#333333",
        width=2,
    )

    draw.text(
        (
            535,
            footer_y + 78,
        ),
        str(
            len(
                sheet_data[
                    "date_colors"
                ]
            )
        ),
        fill="#111111",
        font=footer_value_font,
    )

    # TOTAL CAPACITY
    draw.text(
        (
            930,
            footer_y + 28,
        ),
        "Total Capacity (MW)",
        fill="#111111",
        font=footer_font,
    )

    draw.line(
        (
            930,
            footer_y + 70,
            1320,
            footer_y + 70,
        ),
        fill="#333333",
        width=2,
    )

    if sheet_data[
        "total_capacity"
    ]:

        draw.text(
            (
                930,
                footer_y + 78,
            ),
            sheet_data[
                "total_capacity"
            ],
            fill="#111111",
            font=footer_value_font,
        )

    return image


# ============================================================
# SESSION STATE
# ============================================================

if "active_sheet" not in st.session_state:

    st.session_state.active_sheet = 1


if "selected_day" not in st.session_state:

    st.session_state.selected_day = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("☀️ Solar Sheets")

    st.caption(
        "10 sheets"
    )

    # --------------------------------------------------------
    # SHEET NAVIGATION
    # --------------------------------------------------------

    st.subheader(
        "Select Sheet"
    )

    for sheet_no in range(
        1,
        TOTAL_SHEETS + 1
    ):

        # Show number of colored dates
        sheet_preview = load_sheet(
            sheet_no
        )

        colored_count = len(
            sheet_preview[
                "date_colors"
            ]
        )

        if (
            sheet_no
            == st.session_state.active_sheet
        ):

            button_text = (
                f"🟦 Sheet {sheet_no}"
                f"   ({colored_count})"
            )

        else:

            button_text = (
                f"Sheet {sheet_no}"
                f"   ({colored_count})"
            )

        if st.button(
            button_text,
            key=f"sheet_button_{sheet_no}",
            use_container_width=True,
        ):

            st.session_state.active_sheet = (
                sheet_no
            )

            st.session_state.selected_day = (
                None
            )

            st.rerun()

    # --------------------------------------------------------
    # CURRENT SHEET
    # --------------------------------------------------------

    active_sheet = (
        st.session_state.active_sheet
    )

    current = load_sheet(
        active_sheet
    )

    st.divider()

    st.subheader(
        "Header"
    )

    header_text = st.text_input(
        "Left header",
        value=current[
            "header_text"
        ],
        placeholder="Solar - G1",
        key=f"header_input_{active_sheet}",
    )

    month_year = st.selectbox(
        "Month + Year",
        options=[""] + MONTH_LIST,
        index=(
            (
                [""] + MONTH_LIST
            ).index(
                current[
                    "month_year"
                ]
            )
            if current[
                "month_year"
            ] in MONTH_LIST
            else 0
        ),
        format_func=lambda x:
            "Select Month-Year"
            if x == ""
            else x,
        key=f"month_input_{active_sheet}",
    )

    st.divider()

    st.subheader(
        "Footer"
    )

    mal_capacity = st.text_input(
        "MAL Capacity (MW)",
        value=current[
            "mal_capacity"
        ],
        key=f"mal_input_{active_sheet}",
    )

    total_capacity = st.text_input(
        "Total Capacity (MW)",
        value=current[
            "total_capacity"
        ],
        key=f"total_input_{active_sheet}",
    )

    # --------------------------------------------------------
    # SAVE DETAILS
    # --------------------------------------------------------

    if st.button(
        "💾 Save Sheet Details",
        use_container_width=True,
    ):

        save_sheet_details(
            active_sheet,
            header_text,
            month_year,
            mal_capacity,
            total_capacity,
        )

        st.session_state.selected_day = (
            None
        )

        st.success(
            "Saved"
        )

        st.rerun()

    # --------------------------------------------------------
    # COLOR CONTROL
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Color Date"
    )

    selected_day = (
        st.session_state.selected_day
    )

    if selected_day is None:

        st.info(
            "Click a date below."
        )

    else:

        st.success(
            f"Selected Date: {selected_day}"
        )

        st.write(
            "Choose color:"
        )

        color_columns = st.columns(
            2
        )

        for index, color_name in enumerate(
            [
                "Red",
                "Green",
                "Yellow",
                "Blue",
            ]
        ):

            with color_columns[
                index % 2
            ]:

                if st.button(
                    f"{EMOJIS[color_name]} {color_name}",
                    key=(
                        f"color_"
                        f"{active_sheet}_"
                        f"{selected_day}_"
                        f"{color_name}"
                    ),
                    use_container_width=True,
                ):

                    # ----------------------------------------
                    # SAVE IMMEDIATELY
                    # ----------------------------------------

                    save_date_color(
                        active_sheet,
                        selected_day,
                        color_name,
                    )

                    # ----------------------------------------
                    # FIND NEXT UNCOLORED DATE
                    # ----------------------------------------

                    updated = load_sheet(
                        active_sheet
                    )

                    number_of_days = (
                        days_in_month(
                            updated[
                                "month_year"
                            ]
                        )
                    )

                    next_day = None

                    for candidate in range(
                        selected_day + 1,
                        number_of_days + 1
                    ):

                        if candidate not in (
                            updated[
                                "date_colors"
                            ]
                        ):

                            next_day = candidate
                            break

                    # If no later date exists,
                    # find any uncolored date.
                    if next_day is None:

                        for candidate in range(
                            1,
                            number_of_days + 1
                        ):

                            if candidate not in (
                                updated[
                                    "date_colors"
                                ]
                            ):

                                next_day = candidate
                                break

                    st.session_state.selected_day = (
                        next_day
                    )

                    st.rerun()

        st.divider()

        if st.button(
            "🗑 Clear Date",
            use_container_width=True,
        ):

            delete_date_color(
                active_sheet,
                selected_day,
            )

            st.session_state.selected_day = (
                None
            )

            st.rerun()

        if st.button(
            "✖ Cancel",
            use_container_width=True,
        ):

            st.session_state.selected_day = (
                None
            )

            st.rerun()

    # --------------------------------------------------------
    # CLEAR SHEET
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🗑 Clear All Dates",
        use_container_width=True,
    ):

        clear_sheet_dates(
            active_sheet
        )

        st.session_state.selected_day = (
            None
        )

        st.rerun()


# ============================================================
# LOAD CURRENT SHEET AGAIN
# ============================================================

active_sheet = (
    st.session_state.active_sheet
)

current = load_sheet(
    active_sheet
)

number_of_days = days_in_month(
    current[
        "month_year"
    ]
)


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        border-bottom:3px solid #222;
        padding:10px 8px 12px 8px;
        margin-bottom:15px;
    ">

        <div style="
            font-size:34px;
            font-weight:700;
        ">
            {current["header_text"] or f"Solar - {active_sheet}"}
        </div>

        <div style="
            font-size:29px;
            font-weight:700;
        ">
            {current["month_year"] or "Month-Year"}
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.info(
    "Click a date number → choose Red, Green, Yellow or Blue. "
    "The color is saved immediately. After coloring a date, "
    "the next uncolored date is selected automatically."
)


# ============================================================
# DATE COLOR LEGEND
# ============================================================

legend_columns = st.columns(
    4
)

for column, color_name in zip(
    legend_columns,
    [
        "Red",
        "Green",
        "Yellow",
        "Blue",
    ],
):

    with column:

        st.markdown(
            f"""
            <div style="
                padding:8px;
                border-radius:8px;
                background:{COLORS[color_name]};
                text-align:center;
                font-weight:700;
                color:#111;
            ">
                {EMOJIS[color_name]} {color_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DATE SELECTION AREA
# ============================================================

st.subheader(
    "Calendar / Brain"
)

st.caption(
    "The numbers represent calendar dates."
)


# ============================================================
# BRAIN-LIKE DATE BUTTONS
# ============================================================

# CSS makes the date area visually cleaner.
st.markdown(
    """
    <style>

    .date-help {
        text-align:center;
        font-size:13px;
        color:#666;
    }

    div[data-testid="stButton"] button {
        border-radius:10px;
        min-height:48px;
        font-weight:700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# DATE POSITIONS
# ------------------------------------------------------------

left_rows = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
]

right_rows = [
    [21, 22, 23, 24, 25],
    [26, 27, 28, 29, 30],
    [31],
]


def show_date_button(
    day,
    column,
):

    if day > number_of_days:

        return

    color = current[
        "date_colors"
    ].get(day)

    if color:

        label = (
            f"{EMOJIS[color]} {day}"
        )

    else:

        label = str(day)

    with column:

        if st.button(
            label,
            key=(
                f"date_button_"
                f"{active_sheet}_"
                f"{day}"
            ),
            use_container_width=True,
        ):

            st.session_state.selected_day = (
                day
            )

            st.rerun()


# ------------------------------------------------------------
# BRAIN GRID
# ------------------------------------------------------------

for row_index in range(7):

    columns = st.columns(
        [
            1,
            1,
            1,
            1,
            0.25,
            0.25,
            1,
            1,
            1,
            1,
        ]
    )

    # LEFT SIDE
    if row_index < len(
        left_rows
    ):

        for index, day in enumerate(
            left_rows[row_index]
        ):

            show_date_button(
                day,
                columns[index],
            )

    # RIGHT SIDE
    if row_index < len(
        right_rows
    ):

        for index, day in enumerate(
            right_rows[row_index]
        ):

            show_date_button(
                day,
                columns[
                    6 + index
                ],
            )


# ============================================================
# CURRENT SHEET STATUS
# ============================================================

st.divider()

st.subheader(
    "Current Sheet"
)

colored_dates = current[
    "date_colors"
]

status_columns = st.columns(
    4
)

for column, color_name in zip(
    status_columns,
    [
        "Red",
        "Green",
        "Yellow",
        "Blue",
    ],
):

    with column:

        dates = [
            str(day)
            for day, color
            in sorted(
                colored_dates.items()
            )
            if color == color_name
        ]

        st.markdown(
            f"### {EMOJIS[color_name]} {color_name}"
        )

        if dates:

            st.write(
                ", ".join(dates)
            )

        else:

            st.caption(
                "No dates"
            )


# ============================================================
# CURRENT SHEET PREVIEW
# ============================================================

st.divider()

st.subheader(
    "Sheet Preview"
)

current_image = create_sheet_image(
    active_sheet,
    current,
)

st.image(
    current_image,
    use_container_width=True,
)


# ============================================================
# DOWNLOAD CURRENT SHEET
# ============================================================

current_buffer = io.BytesIO()

current_image.save(
    current_buffer,
    format="PNG"
)

current_buffer.seek(0)

st.download_button(
    "📥 Download Current Sheet",
    data=current_buffer.getvalue(),
    file_name=(
        f"Solar_Sheet_{active_sheet}.png"
    ),
    mime="image/png",
)


# ============================================================
# FINAL 10-SHEET COLLAGE
# ============================================================

st.divider()

st.header(
    "Final 10-Sheet Collage"
)

all_sheets = load_all_sheets()

# ------------------------------------------------------------
# CREATE THUMBNAILS
# ------------------------------------------------------------

thumbnail_width = 680
thumbnail_height = 486

thumbnails = []

for index, sheet_data in enumerate(
    all_sheets,
    start=1,
):

    sheet_image = create_sheet_image(
        index,
        sheet_data,
    )

    sheet_image.thumbnail(
        (
            thumbnail_width,
            thumbnail_height,
        ),
        Image.Resampling.LANCZOS,
    )

    thumbnails.append(
        sheet_image
    )


# ------------------------------------------------------------
# COLLAGE
# ------------------------------------------------------------

columns_count = 2
rows_count = 5

gap = 18

collage_width = (
    columns_count
    * thumbnail_width
    +
    (columns_count + 1)
    * gap
)

collage_height = (
    rows_count
    * thumbnail_height
    +
    (rows_count + 1)
    * gap
)

collage = Image.new(
    "RGB",
    (
        collage_width,
        collage_height,
    ),
    "#DDDDDD",
)


# ------------------------------------------------------------
# PLACE SHEETS
# ------------------------------------------------------------

for index, sheet_image in enumerate(
    thumbnails
):

    row = (
        index
        //
        columns_count
    )

    column = (
        index
        %
        columns_count
    )

    x = (
        gap
        +
        column
        * (
            thumbnail_width
            + gap
        )
    )

    y = (
        gap
        +
        row
        * (
            thumbnail_height
            + gap
        )
    )

    collage.paste(
        sheet_image,
        (
            x,
            y,
        )
    )


st.image(
    collage,
    caption="10-Sheet Collage",
    use_container_width=True,
)


# ============================================================
# DOWNLOAD COLLAGE
# ============================================================

collage_buffer = io.BytesIO()

collage.save(
    collage_buffer,
    format="PNG",
)

collage_buffer.seek(0)

st.download_button(
    "📥 Download 10-Sheet Collage",
    data=collage_buffer.getvalue(),
    file_name="Solar_10_Sheet_Collage.png",
    mime="image/png",
)


# ============================================================
# DATA EXPORT
# ============================================================

st.divider()

st.subheader(
    "Saved Data"
)

rows = []

for sheet_no, sheet_data in enumerate(
    all_sheets,
    start=1,
):

    for day, color in sorted(
        sheet_data[
            "date_colors"
        ].items()
    ):

        rows.append(
            {
                "Sheet": sheet_no,
                "Header": sheet_data[
                    "header_text"
                ],
                "Month-Year": sheet_data[
                    "month_year"
                ],
                "Date": day,
                "Color": color,
            }
        )


if rows:

    import pandas as pd

    export_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        export_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = (
        export_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "📊 Download CSV",
        data=csv_data,
        file_name="Solar_Date_Color_Data.csv",
        mime="text/csv",
    )

else:

    st.info(
        "No colored dates have been saved yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "All date colors are saved immediately to the local database."
)
