# ============================================================
# 7_Brain_Penalty.py
# DATE COLORING / BRAIN SHEET
# ============================================================

import io
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Brain Penalty",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_SHEETS = 10

DB_PATH = Path("brain_penalty.db")

COLORS = {
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

@st.cache_resource
def get_database():

    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
    )

    cursor = conn.cursor()

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

    for sheet_no in range(1, TOTAL_SHEETS + 1):

        cursor.execute(
            """
            INSERT OR IGNORE INTO sheets (sheet_no)
            VALUES (?)
            """,
            (sheet_no,),
        )

    conn.commit()

    return conn


DB = get_database()


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_sheet(sheet_no):

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

        return {
            "header_text": "",
            "month_year": "",
            "footer_left": "",
            "footer_middle": "",
            "footer_right": "",
            "date_colors": {},
        }

    cursor.execute(
        """
        SELECT day, color
        FROM date_colors
        WHERE sheet_no = ?
        ORDER BY day
        """,
        (sheet_no,),
    )

    colors = {
        int(day): color
        for day, color in cursor.fetchall()
    }

    return {
        "header_text": row[0] or "",
        "month_year": row[1] or "",
        "footer_left": row[2] or "",
        "footer_middle": row[3] or "",
        "footer_right": row[4] or "",
        "date_colors": colors,
    }


def save_sheet(
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


def save_color(
    sheet_no,
    day,
    color,
):

    DB.execute(
        """
        INSERT OR REPLACE INTO date_colors
        (sheet_no, day, color)
        VALUES (?, ?, ?)
        """,
        (
            sheet_no,
            day,
            color,
        ),
    )

    DB.commit()


def remove_color(
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


def clear_sheet(
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
# MONTHS
# ============================================================

MONTH_OPTIONS = [""]

for year in range(2024, 2036):

    for month in range(1, 13):

        MONTH_OPTIONS.append(
            datetime(
                year,
                month,
                1,
            ).strftime("%B %Y")
        )


def get_days(
    month_year,
):

    if not month_year:

        return 31

    try:

        dt = datetime.strptime(
            month_year,
            "%B %Y",
        )

        if dt.month == 12:

            nxt = datetime(
                dt.year + 1,
                1,
                1,
            )

        else:

            nxt = datetime(
                dt.year,
                dt.month + 1,
                1,
            )

        return (
            nxt - dt
        ).days

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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Brain Sheets")

    st.caption(
        "Select one of the 10 sheets"
    )

    # --------------------------------------------------------
    # SHEETS
    # --------------------------------------------------------

    for sheet_no in range(
        1,
        TOTAL_SHEETS + 1,
    ):

        data = get_sheet(
            sheet_no
        )

        count = len(
            data["date_colors"]
        )

        if sheet_no == st.session_state.active_sheet:

            label = (
                f"🟦 Sheet {sheet_no} "
                f"• {count} colored"
            )

        else:

            label = (
                f"Sheet {sheet_no} "
                f"• {count} colored"
            )

        if st.button(
            label,
            key=f"sheet_{sheet_no}",
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

    sheet_no = st.session_state.active_sheet

    data = get_sheet(
        sheet_no
    )

    st.divider()

    st.subheader(
        f"Sheet {sheet_no} Details"
    )

    header_text = st.text_input(
        "Left Header",
        value=data["header_text"],
        placeholder="Example: Solar - G1",
        key=f"header_{sheet_no}",
    )

    month_year = st.selectbox(
        "Month & Year",
        MONTH_OPTIONS,
        index=(
            MONTH_OPTIONS.index(
                data["month_year"]
            )
            if data["month_year"]
            in MONTH_OPTIONS
            else 0
        ),
        key=f"month_{sheet_no}",
    )

    st.divider()

    st.subheader(
        "Footer"
    )

    footer_left = st.text_input(
        "Left",
        value=data["footer_left"],
        key=f"footer_left_{sheet_no}",
    )

    footer_middle = st.text_input(
        "Middle",
        value=data["footer_middle"],
        key=f"footer_middle_{sheet_no}",
    )

    footer_right = st.text_input(
        "Right",
        value=data["footer_right"],
        key=f"footer_right_{sheet_no}",
    )

    if st.button(
        "💾 Save Details",
        use_container_width=True,
    ):

        save_sheet(
            sheet_no,
            header_text,
            month_year,
            footer_left,
            footer_middle,
            footer_right,
        )

        st.success(
            "Saved"
        )

        st.rerun()

    # --------------------------------------------------------
    # COLOR SELECTOR
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Date Coloring"
    )

    selected_day = (
        st.session_state.selected_day
    )

    if selected_day is None:

        st.info(
            "Click a date on the sheet."
        )

    else:

        st.success(
            f"Date {selected_day} selected"
        )

        for color in COLORS:

            if st.button(
                f"{COLOR_EMOJI[color]} {color}",
                key=(
                    f"select_color_"
                    f"{sheet_no}_"
                    f"{selected_day}_"
                    f"{color}"
                ),
                use_container_width=True,
            ):

                save_color(
                    sheet_no,
                    selected_day,
                    color,
                )

                # --------------------------------------------
                # AUTOMATICALLY SELECT NEXT UNCOLORED DATE
                # --------------------------------------------

                updated = get_sheet(
                    sheet_no
                )

                max_days = get_days(
                    updated["month_year"]
                )

                next_day = None

                # First search forward
                for candidate in range(
                    selected_day + 1,
                    max_days + 1,
                ):

                    if candidate not in updated[
                        "date_colors"
                    ]:

                        next_day = candidate
                        break

                # Then search from beginning
                if next_day is None:

                    for candidate in range(
                        1,
                        max_days + 1,
                    ):

                        if candidate not in updated[
                            "date_colors"
                        ]:

                            next_day = candidate
                            break

                st.session_state.selected_day = (
                    next_day
                )

                st.rerun()

        if st.button(
            "🗑 Remove Color",
            use_container_width=True,
        ):

            remove_color(
                sheet_no,
                selected_day,
            )

            st.session_state.selected_day = (
                None
            )

            st.rerun()

        if st.button(
            "Cancel",
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

        clear_sheet(
            sheet_no
        )

        st.session_state.selected_day = (
            None
        )

        st.rerun()


# ============================================================
# LOAD CURRENT DATA
# ============================================================

sheet_no = st.session_state.active_sheet

data = get_sheet(
    sheet_no
)

max_days = get_days(
    data["month_year"]
)


# ============================================================
# MAIN HEADER
# ============================================================

header_col, month_col = st.columns(
    [3, 1]
)

with header_col:

    st.title(
        data["header_text"]
        or f"Solar - Sheet {sheet_no}"
    )

with month_col:

    st.markdown(
        f"### {data['month_year'] or 'Month-Year'}"
    )


st.divider()


# ============================================================
# STATUS
# ============================================================

colored = data[
    "date_colors"
]

st.write(
    f"**Sheet {sheet_no}**  •  "
    f"{len(colored)} / {max_days} dates colored"
)


# ============================================================
# COLOR LEGEND
# ============================================================

legend = st.columns(4)

for column, color in zip(
    legend,
    COLORS,
):

    with column:

        st.write(
            f"{COLOR_EMOJI[color]} **{color}**"
        )


# ============================================================
# DATE GRID
# ============================================================

st.subheader(
    "Dates"
)

st.caption(
    "Click any date to select it, then choose its color from the sidebar."
)


# ============================================================
# BRAIN-LIKE GRID
#
# IMPORTANT:
# We use exactly 5 columns on each side.
# No columns[6 + index] access.
# Therefore the previous IndexError cannot occur.
# ============================================================

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
):

    if day > max_days:

        return

    color = colored.get(
        day
    )

    if color:

        label = (
            f"{COLOR_EMOJI[color]} {day}"
        )

    else:

        label = f"Day {day}"

    if st.button(
        label,
        key=(
            f"date_"
            f"{sheet_no}_"
            f"{day}"
        ),
        use_container_width=True,
    ):

        st.session_state.selected_day = (
            day
        )

        st.rerun()


# ============================================================
# LEFT BRAIN
# ============================================================

st.markdown(
    "### Left"
)

for row in left_dates:

    cols = st.columns(
        len(row)
    )

    for col, day in zip(
        cols,
        row,
    ):

        with col:

            date_button(
                day
            )


# ============================================================
# RIGHT BRAIN
# ============================================================

st.markdown(
    "### Right"
)

for row in right_dates:

    cols = st.columns(
        len(row)
    )

    for col, day in zip(
        cols,
        row,
    ):

        with col:

            date_button(
                day
            )


# ============================================================
# SELECTED DATE
# ============================================================

if st.session_state.selected_day is not None:

    st.divider()

    selected = (
        st.session_state.selected_day
    )

    selected_color = colored.get(
        selected
    )

    if selected_color:

        st.success(
            f"Date {selected} is currently "
            f"{selected_color}."
        )

    else:

        st.info(
            f"Date {selected} has no color yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer = st.columns(3)

with footer[0]:

    st.write(
        f"**{data['footer_left']}**"
    )

with footer[1]:

    st.write(
        f"**{data['footer_middle']}**"
    )

with footer[2]:

    st.write(
        f"**{data['footer_right']}**"
    )


# ============================================================
# CURRENT SHEET DATA
# ============================================================

st.divider()

st.subheader(
    "Color Summary"
)

summary_cols = st.columns(4)

for column, color in zip(
    summary_cols,
    COLORS,
):

    dates = [
        day
        for day, c in sorted(
            colored.items()
        )
        if c == color
    ]

    with column:

        st.metric(
            label=color,
            value=len(dates),
        )

        if dates:

            st.caption(
                "Dates: "
                + ", ".join(
                    map(
                        str,
                        dates,
                    )
                )
            )


# ============================================================
# EXPORT DATA
# ============================================================

rows = []

for s in range(
    1,
    TOTAL_SHEETS + 1,
):

    sheet = get_sheet(
        s
    )

    for day, color in sorted(
        sheet["date_colors"].items()
    ):

        rows.append(
            {
                "Sheet": s,
                "Header": sheet[
                    "header_text"
                ],
                "Month-Year": sheet[
                    "month_year"
                ],
                "Date": day,
                "Color": color,
            }
        )


if rows:

    export_df = pd.DataFrame(
        rows
    )

    csv_data = export_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "📥 Download Saved Data",
        data=csv_data,
        file_name="brain_penalty_data.csv",
        mime="text/csv",
    )
