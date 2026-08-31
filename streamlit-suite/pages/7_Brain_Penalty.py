# ============================================================
# 7_Brain_Penalty.py
#
# BRAIN DATE COLORING SHEET
#
# 10 Sheets
# Date based
# Red / Green / Yellow / Blue
# Persistent SQLite storage
# No sidebar
# No slider
# No HTML UI
# ============================================================

import io
import sqlite3
from pathlib import Path
from datetime import datetime
import calendar

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Penalty",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_SHEETS = 10

DB_FILE = Path("brain_penalty.db")

COLOR_NAMES = [
    "Red",
    "Green",
    "Yellow",
    "Blue",
]

COLOR_VALUES = {
    "Red": "#E53935",
    "Green": "#43A047",
    "Yellow": "#FDD835",
    "Blue": "#1E88E5",
}

COLOR_SYMBOLS = {
    "Red": "🔴",
    "Green": "🟢",
    "Yellow": "🟡",
    "Blue": "🔵",
}


# ============================================================
# DATABASE
# ============================================================

@st.cache_resource
def initialize_database():

    conn = sqlite3.connect(
        str(DB_FILE),
        check_same_thread=False,
    )

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
            footer_left TEXT DEFAULT '',
            footer_middle TEXT DEFAULT '',
            footer_right TEXT DEFAULT ''
        )
        """
    )

    # --------------------------------------------------------
    # DATES
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
        TOTAL_SHEETS + 1,
    ):

        cursor.execute(
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


DB = initialize_database()


# ============================================================
# DATABASE HELPERS
# ============================================================

def load_sheet(sheet_no):

    cursor = DB.cursor()

    cursor.execute(
        """
        SELECT
            header_text,
            month_year,
            footer_left,
            footer_middle,
            footer_right
        FROM sheets
        WHERE sheet_no = ?
        """,
        (sheet_no,),
    )

    row = cursor.fetchone()

    if row is None:

        sheet = {
            "header_text": "",
            "month_year": "",
            "footer_left": "",
            "footer_middle": "",
            "footer_right": "",
            "date_colors": {},
        }

        return sheet

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

    color_rows = cursor.fetchall()

    date_colors = {}

    for day, color in color_rows:

        date_colors[int(day)] = color

    return {
        "header_text": row[0] or "",
        "month_year": row[1] or "",
        "footer_left": row[2] or "",
        "footer_middle": row[3] or "",
        "footer_right": row[4] or "",
        "date_colors": date_colors,
    }


def save_sheet_details(
    sheet_no,
    header_text,
    month_year,
    footer_left,
    footer_middle,
    footer_right,
):

    DB.execute(
        """
        UPDATE sheets
        SET
            header_text = ?,
            month_year = ?,
            footer_left = ?,
            footer_middle = ?,
            footer_right = ?
        WHERE sheet_no = ?
        """,
        (
            header_text,
            month_year,
            footer_left,
            footer_middle,
            footer_right,
            sheet_no,
        ),
    )

    DB.commit()


def save_date_color(
    sheet_no,
    day,
    color,
):

    if color not in COLOR_NAMES:
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
            color,
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


def clear_sheet_colors(
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
# MONTH / YEAR OPTIONS
# ============================================================

MONTH_OPTIONS = [""]

for year in range(
    2024,
    2036,
):

    for month in range(
        1,
        13,
    ):

        MONTH_OPTIONS.append(
            datetime(
                year,
                month,
                1,
            ).strftime(
                "%B %Y"
            )
        )


def get_days_in_month(
    month_year,
):

    if not month_year:
        return 31

    try:

        dt = datetime.strptime(
            month_year,
            "%B %Y",
        )

        return calendar.monthrange(
            dt.year,
            dt.month,
        )[1]

    except Exception:

        return 31


# ============================================================
# SESSION STATE
# ============================================================

if "active_sheet" not in st.session_state:

    st.session_state.active_sheet = 1


if "selected_day" not in st.session_state:

    st.session_state.selected_day = None


# ============================================================
# CSS
#
# Only used to make standard Streamlit controls look cleaner.
# No HTML is displayed to the user.
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
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    div.stButton > button {
        min-height: 44px;
        border-radius: 9px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP TITLE
# ============================================================

st.title(
    "🧠 Brain Penalty"
)

st.caption(
    "Daily date coloring sheet"
)


# ============================================================
# SHEET NAVIGATION
# ============================================================

st.subheader(
    "Sheets"
)

sheet_columns = st.columns(
    TOTAL_SHEETS
)

for index in range(
    TOTAL_SHEETS
):

    sheet_no = index + 1

    sheet_data = load_sheet(
        sheet_no
    )

    colored_count = len(
        sheet_data[
            "date_colors"
        ]
    )

    with sheet_columns[index]:

        if sheet_no == st.session_state.active_sheet:

            button_label = (
                f"🟦 {sheet_no}"
            )

        else:

            button_label = (
                f"{sheet_no}"
            )

        if st.button(
            button_label,
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

        st.caption(
            f"{colored_count} colored"
        )


# ============================================================
# CURRENT SHEET
# ============================================================

active_sheet = (
    st.session_state.active_sheet
)

sheet = load_sheet(
    active_sheet
)


# ============================================================
# HEADER INPUT
# ============================================================

st.divider()

header_left, header_right = st.columns(
    [3, 1]
)

with header_left:

    header_text = st.text_input(
        "Left Header",
        value=sheet[
            "header_text"
        ],
        placeholder="Enter text...",
        key=f"header_input_{active_sheet}",
    )

with header_right:

    current_month_index = 0

    if sheet[
        "month_year"
    ] in MONTH_OPTIONS:

        current_month_index = (
            MONTH_OPTIONS.index(
                sheet[
                    "month_year"
                ]
            )
        )

    month_year = st.selectbox(
        "Month & Year",
        MONTH_OPTIONS,
        index=current_month_index,
        key=f"month_input_{active_sheet}",
        format_func=lambda x:
            "Select Month & Year"
            if not x
            else x,
    )


# ============================================================
# FOOTER INPUTS
# ============================================================

footer_left, footer_middle, footer_right = st.columns(
    3
)

with footer_left:

    footer_left_value = st.text_input(
        "Footer Left",
        value=sheet[
            "footer_left"
        ],
        key=f"footer_left_{active_sheet}",
    )

with footer_middle:

    footer_middle_value = st.text_input(
        "Footer Middle",
        value=sheet[
            "footer_middle"
        ],
        key=f"footer_middle_{active_sheet}",
    )

with footer_right:

    footer_right_value = st.text_input(
        "Footer Right",
        value=sheet[
            "footer_right"
        ],
        key=f"footer_right_{active_sheet}",
    )


# ============================================================
# SAVE HEADER / FOOTER
# ============================================================

save_col, status_col = st.columns(
    [1, 4]
)

with save_col:

    if st.button(
        "💾 Save Details",
        use_container_width=True,
    ):

        save_sheet_details(
            active_sheet,
            header_text,
            month_year,
            footer_left_value,
            footer_middle_value,
            footer_right_value,
        )

        st.success(
            "Saved"
        )

        st.rerun()


# ============================================================
# RELOAD DATA AFTER SAVE
# ============================================================

sheet = load_sheet(
    active_sheet
)

max_days = get_days_in_month(
    sheet[
        "month_year"
    ]
)

date_colors = sheet[
    "date_colors"
]


# ============================================================
# VISUAL SHEET HEADER
# ============================================================

st.divider()

visual_header_left, visual_header_right = st.columns(
    [3, 1]
)

with visual_header_left:

    st.markdown(
        f"## {sheet['header_text'] or 'Solar'}"
    )

with visual_header_right:

    st.markdown(
        f"## {sheet['month_year'] or 'Month Year'}"
    )


# ============================================================
# BRAIN DATE LAYOUT
# ============================================================

st.markdown(
    "### 🧠 Date Sheet"
)

st.caption(
    "Click any date. Then choose Red, Green, Yellow or Blue."
)


# ============================================================
# BRAIN LAYOUT
#
# The layout deliberately uses fixed row structures.
# There is NO calculated columns[index] access.
# ============================================================

LEFT_ROWS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23],
]

RIGHT_ROWS = [
    [24, 25, 26, 27],
    [28, 29, 30],
    [31],
]


def draw_date_button(
    day,
):

    if day > max_days:
        return

    current_color = date_colors.get(
        day
    )

    if current_color:

        label = (
            f"{COLOR_SYMBOLS[current_color]} "
            f"{day}"
        )

    else:

        label = str(day)

    if st.button(
        label,
        key=f"date_{active_sheet}_{day}",
        use_container_width=True,
    ):

        st.session_state.selected_day = (
            day
        )

        st.rerun()


# ============================================================
# BRAIN TOP
# ============================================================

st.markdown(
    "#### Left Hemisphere"
)

for row in LEFT_ROWS:

    cols = st.columns(
        len(row)
    )

    for col, day in zip(
        cols,
        row,
    ):

        with col:

            draw_date_button(
                day
            )


st.markdown(
    "#### Right Hemisphere"
)

for row in RIGHT_ROWS:

    cols = st.columns(
        len(row)
    )

    for col, day in zip(
        cols,
        row,
    ):

        with col:

            draw_date_button(
                day
            )


# ============================================================
# SELECTED DATE
# ============================================================

selected_day = (
    st.session_state.selected_day
)

if selected_day is not None:

    st.divider()

    st.subheader(
        f"Selected Date: {selected_day}"
    )

    selected_color = date_colors.get(
        selected_day
    )

    if selected_color:

        st.write(
            f"Current color: "
            f"{COLOR_SYMBOLS[selected_color]} "
            f"{selected_color}"
        )

    else:

        st.write(
            "Current color: Not colored"
        )

    st.write(
        "Choose a color:"
    )

    color_columns = st.columns(
        4
    )

    for col, color_name in zip(
        color_columns,
        COLOR_NAMES,
    ):

        with col:

            if st.button(
                f"{COLOR_SYMBOLS[color_name]} {color_name}",
                key=(
                    f"color_"
                    f"{active_sheet}_"
                    f"{selected_day}_"
                    f"{color_name}"
                ),
                use_container_width=True,
            ):

                save_date_color(
                    active_sheet,
                    selected_day,
                    color_name,
                )

                # --------------------------------------------
                # FIND NEXT UNCOLORED DATE
                # --------------------------------------------

                updated = load_sheet(
                    active_sheet
                )

                updated_colors = updated[
                    "date_colors"
                ]

                updated_max_days = (
                    get_days_in_month(
                        updated[
                            "month_year"
                        ]
                    )
                )

                next_day = None

                # Search forward
                for candidate in range(
                    selected_day + 1,
                    updated_max_days + 1,
                ):

                    if candidate not in updated_colors:

                        next_day = candidate

                        break

                # If nothing after it,
                # search from the beginning.
                if next_day is None:

                    for candidate in range(
                        1,
                        updated_max_days + 1,
                    ):

                        if candidate not in updated_colors:

                            next_day = candidate

                            break

                st.session_state.selected_day = (
                    next_day
                )

                st.rerun()


    delete_col, cancel_col = st.columns(
        2
    )

    with delete_col:

        if st.button(
            "🗑 Remove Color",
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

    with cancel_col:

        if st.button(
            "Cancel Selection",
            use_container_width=True,
        ):

            st.session_state.selected_day = (
                None
            )

            st.rerun()


# ============================================================
# COLOR SUMMARY
# ============================================================

st.divider()

st.subheader(
    "Current Sheet Summary"
)

summary_columns = st.columns(
    4
)

for col, color_name in zip(
    summary_columns,
    COLOR_NAMES,
):

    dates = [
        day
        for day, color
        in sorted(
            date_colors.items()
        )
        if color == color_name
    ]

    with col:

        st.metric(
            color_name,
            len(dates),
        )

        if dates:

            st.caption(
                "Dates: "
                + ", ".join(
                    str(x)
                    for x in dates
                )
            )


# ============================================================
# FOOTER PREVIEW
# ============================================================

st.divider()

footer_preview = st.columns(
    3
)

with footer_preview[0]:

    st.write(
        f"**{sheet['footer_left']}**"
    )

with footer_preview[1]:

    st.write(
        f"**{sheet['footer_middle']}**"
    )

with footer_preview[2]:

    st.write(
        f"**{sheet['footer_right']}**"
    )


# ============================================================
# ALL SHEETS DATA
# ============================================================

st.divider()

st.subheader(
    "Saved Data"
)

all_rows = []

for sheet_no in range(
    1,
    TOTAL_SHEETS + 1,
):

    sheet_data = load_sheet(
        sheet_no
    )

    for day, color in sorted(
        sheet_data[
            "date_colors"
        ].items()
    ):

        all_rows.append(
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


if all_rows:

    df = pd.DataFrame(
        all_rows
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    csv_file = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "📥 Download CSV",
        data=csv_file,
        file_name="brain_penalty_dates.csv",
        mime="text/csv",
    )

else:

    st.info(
        "No dates have been colored yet."
    )


# ============================================================
# RESET ALL DATA
# ============================================================

st.divider()

with st.expander(
    "⚠️ Reset Current Sheet"
):

    st.warning(
        f"This will remove all colors from Sheet {active_sheet}."
    )

    if st.button(
        "Reset Current Sheet",
        key=f"reset_{active_sheet}",
    ):

        clear_sheet_colors(
            active_sheet
        )

        st.session_state.selected_day = (
            None
        )

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 Brain Penalty • Date colors are saved automatically."
)
