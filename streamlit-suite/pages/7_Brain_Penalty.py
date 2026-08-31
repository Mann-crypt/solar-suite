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
    page_title="Solar Date Brain",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DB_PATH = Path("solar_brain.db")

TOTAL_SHEETS = 10

COLOR_MAP = {
    "Red": "#E53935",
    "Green": "#43A047",
    "Yellow": "#FDD835",
    "Blue": "#1E88E5",
}

COLOR_EMOJI = {
    "Red": "🔴",
    "Green": "🟢",
    "Yellow": "🟡",
    "Blue": "🔵",
}


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    """
    Open SQLite database.

    check_same_thread=False keeps this safe for Streamlit's
    execution model.
    """
    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )


def initialize_database():
    """
    Create all required tables.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # SHEETS
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
    # DATE COLORS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS date_colors (
            sheet_no INTEGER NOT NULL,
            day INTEGER NOT NULL,
            color TEXT NOT NULL,
            PRIMARY KEY (sheet_no, day)
        )
        """
    )

    # --------------------------------------------------------
    # GUARANTEE 10 SHEETS
    # --------------------------------------------------------

    for sheet_no in range(1, TOTAL_SHEETS + 1):

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
# DATABASE HELPERS
# ============================================================

def load_sheet(sheet_no):
    """
    Load one complete sheet.
    """

    conn = get_connection()
    cursor = conn.cursor()

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

    if row is None:

        header_text = ""
        month_year = ""
        mal_capacity = ""
        total_capacity = ""

    else:

        header_text = row[0] or ""
        month_year = row[1] or ""
        mal_capacity = row[2] or ""
        total_capacity = row[3] or ""

    # --------------------------------------------------------
    # LOAD DATE COLORS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT day, color
        FROM date_colors
        WHERE sheet_no = ?
        ORDER BY day
        """,
        (sheet_no,)
    )

    rows = cursor.fetchall()

    date_colors = {
        int(day): color
        for day, color in rows
    }

    conn.close()

    return {
        "header_text": header_text,
        "month_year": month_year,
        "mal_capacity": mal_capacity,
        "total_capacity": total_capacity,
        "date_colors": date_colors,
    }


def save_sheet_details(
    sheet_no,
    header_text,
    month_year,
    mal_capacity,
    total_capacity
):
    """
    Save header/footer information.
    """

    conn = get_connection()

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


def save_date_color(
    sheet_no,
    day,
    color
):
    """
    Save or replace one date's color.
    """

    if color not in COLOR_MAP:
        return

    conn = get_connection()

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


def delete_date_color(
    sheet_no,
    day
):
    """
    Remove color from one date.
    """

    conn = get_connection()

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


def clear_all_dates(sheet_no):
    """
    Clear all date colors for one sheet.
    """

    conn = get_connection()

    conn.execute(
        """
        DELETE FROM date_colors
        WHERE sheet_no = ?
        """,
        (sheet_no,)
    )

    conn.commit()
    conn.close()


def load_all_sheets():
    """
    Load all 10 sheets.
    """

    return [
        load_sheet(sheet_no)
        for sheet_no in range(
            1,
            TOTAL_SHEETS + 1
        )
    ]


# ============================================================
# MONTH / YEAR
# ============================================================

def get_month_year_options():

    months = []

    for year in range(2024, 2036):

        for month in range(1, 13):

            value = datetime(
                year,
                month,
                1
            ).strftime("%b-%Y")

            months.append(value)

    return months


MONTH_YEAR_OPTIONS = get_month_year_options()


def get_number_of_days(month_year):

    """
    Return number of days in selected month.
    """

    if not month_year:
        return 31

    try:

        dt = datetime.strptime(
            month_year,
            "%b-%Y"
        )

        if dt.month == 12:

            next_month = datetime(
                dt.year + 1,
                1,
                1
            )

        else:

            next_month = datetime(
                dt.year,
                dt.month + 1,
                1
            )

        current = datetime(
            dt.year,
            dt.month,
            1
        )

        return (
            next_month - current
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

    candidates = []

    if bold:

        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
            ]
        )

    else:

        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "C:/Windows/Fonts/arial.ttf",
            ]
        )

    for path in candidates:

        if Path(path).exists():

            return ImageFont.truetype(
                path,
                size
            )

    return ImageFont.load_default()


# ============================================================
# DRAW BRAIN SHEET
# ============================================================

def draw_brain_sheet(
    sheet_no,
    sheet_data
):
    """
    Create a printable sheet.

    This is intentionally generated as an image so the final
    10-sheet collage remains stable and predictable.
    """

    WIDTH = 1400
    HEIGHT = 1000

    image = Image.new(
        "RGB",
        (
            WIDTH,
            HEIGHT
        ),
        "white"
    )

    draw = ImageDraw.Draw(image)

    # --------------------------------------------------------
    # FONTS
    # --------------------------------------------------------

    header_font = get_font(
        42,
        True
    )

    month_font = get_font(
        36,
        True
    )

    date_font = get_font(
        25,
        True
    )

    footer_font = get_font(
        24,
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
            HEIGHT - 20
        ),
        radius=18,
        outline="#222222",
        width=4
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
            60,
            52
        ),
        header_text,
        fill="#111111",
        font=header_font
    )

    bbox = draw.textbbox(
        (
            0,
            0
        ),
        month_year,
        font=month_font
    )

    month_width = (
        bbox[2] - bbox[0]
    )

    draw.text(
        (
            WIDTH - month_width - 60,
            58
        ),
        month_year,
        fill="#111111",
        font=month_font
    )

    draw.line(
        (
            55,
            125,
            WIDTH - 55,
            125
        ),
        fill="#333333",
        width=3
    )

    # --------------------------------------------------------
    # BRAIN OUTLINE
    # --------------------------------------------------------

    # Left lobe
    left_brain = (
        70,
        170,
        690,
        790
    )

    # Right lobe
    right_brain = (
        710,
        170,
        1330,
        790
    )

    draw.ellipse(
        left_brain,
        outline="#333333",
        width=5
    )

    draw.ellipse(
        right_brain,
        outline="#333333",
        width=5
    )

    # --------------------------------------------------------
    # CENTRAL BRAIN GROOVE
    # --------------------------------------------------------

    draw.line(
        (
            WIDTH // 2,
            195,
            WIDTH // 2,
            765
        ),
        fill="white",
        width=18
    )

    draw.line(
        (
            WIDTH // 2,
            195,
            WIDTH // 2,
            765
        ),
        fill="#333333",
        width=4
    )

    # --------------------------------------------------------
    # BRAIN FOLDS
    # --------------------------------------------------------

    fold_color = "#8A8A8A"

    folds_left = [
        (120, 250, 350, 315),
        (360, 225, 600, 300),
        (100, 385, 330, 450),
        (365, 365, 610, 440),
        (125, 525, 345, 600),
        (370, 510, 610, 590),
        (190, 665, 450, 720),
    ]

    folds_right = [
        (800, 250, 1030, 315),
        (1050, 225, 1290, 300),
        (790, 385, 1020, 450),
        (1050, 365, 1295, 440),
        (790, 525, 1015, 600),
        (1050, 510, 1290, 590),
        (930, 665, 1190, 720),
    ]

    for box in folds_left:

        draw.arc(
            box,
            180,
            355,
            fill=fold_color,
            width=3
        )

    for box in folds_right:

        draw.arc(
            box,
            185,
            360,
            fill=fold_color,
            width=3
        )

    # --------------------------------------------------------
    # DATE POSITIONS
    # --------------------------------------------------------

    positions = []

    # Left side
    left_rows = [
        (5, 145, 275),
        (5, 145, 385),
        (5, 145, 495),
        (5, 145, 605),
        (4, 190, 700),
    ]

    day = 1

    for count, start_x, y in left_rows:

        for j in range(count):

            positions.append(
                (
                    day,
                    start_x + j * 100,
                    y
                )
            )

            day += 1

    # Right side
    right_rows = [
        (5, 765, 275),
        (5, 765, 385),
        (5, 765, 495),
        (5, 765, 605),
        (4, 810, 700),
    ]

    for count, start_x, y in right_rows:

        for j in range(count):

            positions.append(
                (
                    day,
                    start_x + j * 100,
                    y
                )
            )

            day += 1

    # --------------------------------------------------------
    # DRAW DATES
    # --------------------------------------------------------

    for day, cx, cy in positions:

        selected_color = (
            sheet_data["date_colors"]
            .get(day)
        )

        if selected_color in COLOR_MAP:

            fill = COLOR_MAP[
                selected_color
            ]

        else:

            fill = "#FFFFFF"

        box = (
            cx - 38,
            cy - 30,
            cx + 38,
            cy + 30
        )

        draw.rounded_rectangle(
            box,
            radius=10,
            fill=fill,
            outline="#222222",
            width=3
        )

        text = str(day)

        bbox = draw.textbbox(
            (
                0,
                0
            ),
            text,
            font=date_font
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        text_height = (
            bbox[3] - bbox[1]
        )

        draw.text(
            (
                cx - text_width / 2,
                cy - text_height / 2 - 2
            ),
            text,
            fill="#111111",
            font=date_font
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
            footer_y
        ),
        fill="#333333",
        width=3
    )

    # MAL
    draw.text(
        (
            70,
            footer_y + 25
        ),
        "MAL Capacity (MW)",
        fill="#111111",
        font=footer_font
    )

    draw.line(
        (
            70,
            footer_y + 70,
            430,
            footer_y + 70
        ),
        fill="#333333",
        width=2
    )

    if sheet_data["mal_capacity"]:

        draw.text(
            (
                70,
                footer_y + 78
            ),
            sheet_data["mal_capacity"],
            fill="#111111",
            font=footer_value_font
        )

    # Number of dates
    draw.text(
        (
            535,
            footer_y + 25
        ),
        "No. of Dates",
        fill="#111111",
        font=footer_font
    )

    draw.line(
        (
            535,
            footer_y + 70,
            850,
            footer_y + 70
        ),
        fill="#333333",
        width=2
    )

    colored_dates = len(
        sheet_data["date_colors"]
    )

    draw.text(
        (
            535,
            footer_y + 78
        ),
        str(colored_dates),
        fill="#111111",
        font=footer_value_font
    )

    # Total capacity
    draw.text(
        (
            930,
            footer_y + 25
        ),
        "Total Capacity (MW)",
        fill="#111111",
        font=footer_font
    )

    draw.line(
        (
            930,
            footer_y + 70,
            1320,
            footer_y + 70
        ),
        fill="#333333",
        width=2
    )

    if sheet_data["total_capacity"]:

        draw.text(
            (
                930,
                footer_y + 78
            ),
            sheet_data["total_capacity"],
            fill="#111111",
            font=footer_value_font
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
# HEADER
# ============================================================

st.title("☀️ Solar Date Coloring")

st.caption(
    "Select a date, choose Red / Green / Yellow / Blue, "
    "and the change is saved immediately."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Sheet")

    selected_sheet = st.selectbox(
        "Select sheet",
        options=list(
            range(
                1,
                TOTAL_SHEETS + 1
            )
        ),
        index=(
            st.session_state.active_sheet - 1
        ),
        format_func=lambda x: f"Sheet {x}",
        key="sheet_selector",
    )

    if (
        selected_sheet
        != st.session_state.active_sheet
    ):

        st.session_state.active_sheet = (
            selected_sheet
        )

        st.session_state.selected_day = None

        st.rerun()

    # --------------------------------------------------------
    # LOAD CURRENT SHEET
    # --------------------------------------------------------

    current_sheet = load_sheet(
        st.session_state.active_sheet
    )

    st.divider()

    st.header("Header")

    header_text = st.text_input(
        "Left side",
        value=current_sheet["header_text"],
        placeholder="Solar - G1",
        key=f"header_{selected_sheet}",
    )

    st.markdown("**Month + Year**")

    month_year = st.selectbox(
        "Month-Year",
        options=[""] + MONTH_YEAR_OPTIONS,
        index=(
            ([""] + MONTH_YEAR_OPTIONS).index(
                current_sheet["month_year"]
            )
            if current_sheet["month_year"]
            in MONTH_YEAR_OPTIONS
            else 0
        ),
        format_func=lambda x: (
            "Select month and year"
            if x == ""
            else x
        ),
        key=f"month_{selected_sheet}",
    )

    st.divider()

    st.header("Footer")

    mal_capacity = st.text_input(
        "MAL Capacity (MW)",
        value=current_sheet["mal_capacity"],
        key=f"mal_{selected_sheet}",
    )

    total_capacity = st.text_input(
        "Total Capacity (MW)",
        value=current_sheet["total_capacity"],
        key=f"total_{selected_sheet}",
    )

    # --------------------------------------------------------
    # SAVE HEADER
    # --------------------------------------------------------

    if st.button(
        "💾 Save Sheet Details",
        use_container_width=True,
    ):

        save_sheet_details(
            selected_sheet,
            header_text,
            month_year,
            mal_capacity,
            total_capacity,
        )

        st.success(
            "Sheet details saved."
        )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # COLOR PANEL
    # --------------------------------------------------------

    st.header("Date Coloring")

    selected_day = st.session_state.selected_day

    if selected_day is None:

        st.info(
            "Click a date from the sheet."
        )

    else:

        st.success(
            f"Selected date: {selected_day}"
        )

        st.write(
            "Choose color:"
        )

        for color_name in [
            "Red",
            "Green",
            "Yellow",
            "Blue",
        ]:

            if st.button(
                f"{COLOR_EMOJI[color_name]} {color_name}",
                key=f"color_{selected_sheet}_{selected_day}_{color_name}",
                use_container_width=True,
            ):

                save_date_color(
                    selected_sheet,
                    selected_day,
                    color_name,
                )

                st.session_state.selected_day = None

                st.rerun()

        st.divider()

        if st.button(
            "🗑 Clear This Date",
            use_container_width=True,
        ):

            delete_date_color(
                selected_sheet,
                selected_day,
            )

            st.session_state.selected_day = None

            st.rerun()

        if st.button(
            "✖ Cancel",
            use_container_width=True,
        ):

            st.session_state.selected_day = None

            st.rerun()

    st.divider()

    if st.button(
        "🗑 Clear Current Sheet",
        use_container_width=True,
    ):

        clear_all_dates(
            selected_sheet
        )

        st.session_state.selected_day = None

        st.rerun()


# ============================================================
# RELOAD CURRENT DATA FROM DATABASE
# ============================================================

current_sheet = load_sheet(
    selected_sheet
)

days_in_month = get_number_of_days(
    current_sheet["month_year"]
)


# ============================================================
# SHEET TITLE
# ============================================================

title_left = (
    current_sheet["header_text"]
    or f"Solar - {selected_sheet}"
)

title_right = (
    current_sheet["month_year"]
    or "Month-Year"
)

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        border-bottom:2px solid #333;
        padding:8px 12px;
        margin-bottom:12px;
    ">
        <div style="
            font-size:30px;
            font-weight:700;
        ">
            {title_left}
        </div>

        <div style="
            font-size:25px;
            font-weight:700;
        ">
            {title_right}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BRAIN-LIKE DATE MAP
# ============================================================

st.markdown(
    """
    <style>

    .brain-wrapper {
        display:flex;
        justify-content:center;
        width:100%;
        margin-top:10px;
        margin-bottom:20px;
    }

    .brain {
        width:900px;
        min-height:600px;

        border:5px solid #333;

        border-radius:
            48% 52%
            48% 52%
            /
            50% 48%
            52% 50%;

        padding:45px;

        position:relative;

        background:white;

        box-sizing:border-box;

        box-shadow:
            0 2px 8px
            rgba(0,0,0,0.10);
    }

    .brain-center {
        position:absolute;

        top:30px;
        bottom:30px;

        left:50%;

        border-left:
            5px solid #333;

        transform:
            translateX(-50%);
    }

    .brain-fold {
        position:absolute;

        width:180px;
        height:55px;

        border-top:
            3px solid #999;

        border-radius:50%;

        opacity:0.6;
    }

    .fold-1 {
        left:70px;
        top:90px;
        transform:rotate(12deg);
    }

    .fold-2 {
        left:270px;
        top:165px;
        transform:rotate(-10deg);
    }

    .fold-3 {
        left:80px;
        top:280px;
        transform:rotate(8deg);
    }

    .fold-4 {
        left:280px;
        top:355px;
        transform:rotate(-12deg);
    }

    .fold-5 {
        right:70px;
        top:90px;
        transform:rotate(-12deg);
    }

    .fold-6 {
        right:270px;
        top:165px;
        transform:rotate(10deg);
    }

    .fold-7 {
        right:80px;
        top:280px;
        transform:rotate(-8deg);
    }

    .fold-8 {
        right:280px;
        top:355px;
        transform:rotate(12deg);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Brain outline decoration
# ------------------------------------------------------------

st.markdown(
    """
    <div class="brain-wrapper">
        <div class="brain">

            <div class="brain-center"></div>

            <div class="brain-fold fold-1"></div>
            <div class="brain-fold fold-2"></div>
            <div class="brain-fold fold-3"></div>
            <div class="brain-fold fold-4"></div>

            <div class="brain-fold fold-5"></div>
            <div class="brain-fold fold-6"></div>
            <div class="brain-fold fold-7"></div>
            <div class="brain-fold fold-8"></div>

        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATE BUTTON GRID
# ============================================================

st.subheader(
    "Click a date"
)

st.caption(
    "Colored dates are already saved. "
    "Click any date to change its color."
)


# ============================================================
# DATE LAYOUT
# ============================================================

# We use 10 columns.
#
# 1-5 rows on left
# 16-31 continue on right.
#
# This gives a brain-like visual grouping while retaining
# very reliable native Streamlit buttons.

date_numbers = list(
    range(
        1,
        days_in_month + 1
    )
)


# ------------------------------------------------------------
# Create 5 visual rows
# ------------------------------------------------------------

left_dates = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
]

right_dates = [
    [21, 22, 23, 24, 25],
    [26, 27, 28, 29, 30],
    [31],
]


def date_button(
    day,
    column
):

    if day > days_in_month:

        return

    current_color = (
        current_sheet["date_colors"]
        .get(day)
    )

    if current_color:

        label = (
            f"{COLOR_EMOJI[current_color]} "
            f"{day}"
        )

    else:

        label = str(day)

    if st.button(
        label,
        key=f"date_{selected_sheet}_{day}",
        use_container_width=True,
    ):

        st.session_state.selected_day = day

        st.rerun()


# ============================================================
# BRAIN DATE AREA
# ============================================================

# We use 10 columns, with the middle two columns acting as
# the brain groove.

for row_index in range(7):

    columns = st.columns(
        [
            1,
            1,
            1,
            1,
            0.35,
            0.35,
            1,
            1,
            1,
            1,
        ]
    )

    # Left side
    if row_index < len(left_dates):

        for j, day in enumerate(
            left_dates[row_index]
        ):

            date_button(
                day,
                columns[j]
            )

    # Right side
    if row_index < len(right_dates):

        for j, day in enumerate(
            right_dates[row_index]
        ):

            date_button(
                day,
                columns[6 + j]
            )


# ============================================================
# CURRENT STATUS
# ============================================================

st.divider()

st.subheader(
    f"Sheet {selected_sheet} Status"
)

colored = current_sheet[
    "date_colors"
]

if colored:

    status_cols = st.columns(4)

    for index, color_name in enumerate(
        [
            "Red",
            "Green",
            "Yellow",
            "Blue",
        ]
    ):

        dates = [
            str(day)
            for day, color
            in sorted(
                colored.items()
            )
            if color == color_name
        ]

        with status_cols[index]:

            st.markdown(
                f"### {COLOR_EMOJI[color_name]} {color_name}"
            )

            if dates:

                st.write(
                    ", ".join(dates)
                )

            else:

                st.caption(
                    "None"
                )

else:

    st.info(
        "No dates have been colored yet."
    )


# ============================================================
# PREVIEW
# ============================================================

st.divider()

st.subheader(
    "Sheet Preview"
)

preview = draw_brain_sheet(
    selected_sheet,
    current_sheet
)

st.image(
    preview,
    use_container_width=True
)


# ============================================================
# DOWNLOAD CURRENT SHEET
# ============================================================

preview_buffer = io.BytesIO()

preview.save(
    preview_buffer,
    format="PNG"
)

preview_buffer.seek(0)

st.download_button(
    "📥 Download Current Sheet",
    data=preview_buffer.getvalue(),
    file_name=(
        f"sheet_{selected_sheet}.png"
    ),
    mime="image/png",
)


# ============================================================
# 10-SHEET COLLAGE
# ============================================================

st.divider()

st.header(
    "10-Sheet Final Collage"
)

all_sheets = load_all_sheets()


# ------------------------------------------------------------
# Generate all 10
# ------------------------------------------------------------

sheet_images = []

for index, sheet in enumerate(
    all_sheets,
    start=1
):

    sheet_image = draw_brain_sheet(
        index,
        sheet
    )

    # Resize for collage
    sheet_image = sheet_image.resize(
        (
            680,
            486
        ),
        Image.Resampling.LANCZOS
    )

    sheet_images.append(
        sheet_image
    )


# ------------------------------------------------------------
# 2 x 5 layout
# ------------------------------------------------------------

COLLAGE_COLUMNS = 2
COLLAGE_ROWS = 5

CELL_WIDTH = 680
CELL_HEIGHT = 486

GAP = 18

collage_width = (
    COLLAGE_COLUMNS * CELL_WIDTH
    +
    (COLLAGE_COLUMNS + 1) * GAP
)

collage_height = (
    COLLAGE_ROWS * CELL_HEIGHT
    +
    (COLLAGE_ROWS + 1) * GAP
)

collage = Image.new(
    "RGB",
    (
        collage_width,
        collage_height
    ),
    "#D8D8D8"
)


for index, sheet_image in enumerate(
    sheet_images
):

    row = (
        index
        //
        COLLAGE_COLUMNS
    )

    column = (
        index
        %
        COLLAGE_COLUMNS
    )

    x = (
        GAP
        +
        column * (
            CELL_WIDTH + GAP
        )
    )

    y = (
        GAP
        +
        row * (
            CELL_HEIGHT + GAP
        )
    )

    collage.paste(
        sheet_image,
        (
            x,
            y
        )
    )


st.image(
    collage,
    caption="10-Sheet Collage",
    use_container_width=True
)


# ============================================================
# DOWNLOAD COLLAGE
# ============================================================

collage_buffer = io.BytesIO()

collage.save(
    collage_buffer,
    format="PNG"
)

collage_buffer.seek(0)

st.download_button(
    "📥 Download 10-Sheet Collage",
    data=collage_buffer.getvalue(),
    file_name="solar_10_sheet_collage.png",
    mime="image/png",
)


# ============================================================
# DATABASE BACKUP
# ============================================================

st.divider()

st.subheader(
    "Database"
)

st.caption(
    "Every color selection is written to the database immediately. "
    "Reloading the Streamlit page will reload the saved dates."
)

# ------------------------------------------------------------
# Export CSV
# ------------------------------------------------------------

rows = []

for sheet_no, sheet in enumerate(
    all_sheets,
    start=1
):

    for day, color in sorted(
        sheet["date_colors"].items()
    ):

        rows.append(
            {
                "Sheet": sheet_no,
                "Header": sheet["header_text"],
                "Month-Year": sheet["month_year"],
                "Date": day,
                "Color": color,
            }
        )


if rows:

    import pandas as pd

    export_df = pd.DataFrame(
        rows
    )

    csv_data = export_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "📊 Download Date-Color CSV",
        data=csv_data,
        file_name="solar_date_colors.csv",
        mime="text/csv",
    )

else:

    st.caption(
        "No date-color data available yet."
    )
