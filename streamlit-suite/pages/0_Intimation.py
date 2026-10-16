# --------------------------------------------------
# COMPACT INTIMATION CARDS
# --------------------------------------------------
st.markdown("---")
st.subheader("📋 Intimation Records")

st.caption(
    f"{len(display_df):,} matching records out of {len(df):,} total records"
)

# Card styling
st.markdown(
    """
    <style>
    .record-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        background: var(--secondary-background-color);
        min-height: 210px;
    }

    .record-title {
        font-size: 1.05rem;
        font-weight: 750;
        overflow-wrap: anywhere;
        margin-bottom: 5px;
    }

    .record-state {
        font-size: 0.83rem;
        opacity: 0.75;
        margin-bottom: 12px;
    }

    .record-status {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(100, 116, 139, 0.15);
        overflow-wrap: anywhere;
        max-width: 100%;
    }

    .record-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 12px;
    }

    .record-label {
        font-size: 0.72rem;
        opacity: 0.7;
        margin-bottom: 3px;
    }

    .record-value {
        font-size: 0.88rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .record-coordinates {
        font-size: 0.78rem;
        opacity: 0.8;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value):
    if pd.isna(value):
        return "—"
    value = str(value).strip()
    return value if value else "—"


def make_card(row):
    pss = safe_text(row["PSS"])
    state = safe_text(row["State"])
    capacity = safe_text(row["Capacity (MW)"])
    priority = safe_text(row["Priority"])
    poc1 = safe_text(row["POC1"])
    poc2 = safe_text(row["POC2"])
    poc3 = safe_text(row["POC3"])
    status = safe_text(row["Check Status"])
    lat = safe_text(row["Lat"])
    long = safe_text(row["Long"])

    # Escape source text so Excel values cannot break the card HTML.
    from html import escape

    pss = escape(pss)
    state = escape(state)
    capacity = escape(capacity)
    priority = escape(priority)
    poc1 = escape(poc1)
    poc2 = escape(poc2)
    poc3 = escape(poc3)
    status = escape(status)
    lat = escape(lat)
    long = escape(long)

    return f"""
    <div class="record-card">
        <div class="record-title">⚡ {pss}</div>
        <div class="record-state">📍 {state}</div>
        <span class="record-status">{status}</span>

        <div class="record-grid">
            <div>
                <div class="record-label">Capacity</div>
                <div class="record-value">{capacity} MW</div>
            </div>
            <div>
                <div class="record-label">Priority</div>
                <div class="record-value">{priority}</div>
            </div>
            <div>
                <div class="record-label">POC1</div>
                <div class="record-value">{poc1}</div>
            </div>
            <div>
                <div class="record-label">POC2</div>
                <div class="record-value">{poc2}</div>
            </div>
            <div>
                <div class="record-label">POC3</div>
                <div class="record-value">{poc3}</div>
            </div>
            <div>
                <div class="record-label">Coordinates</div>
                <div class="record-coordinates">{lat}, {long}</div>
            </div>
        </div>
    </div>
    """


if display_df.empty:
    st.info(
        "No matching records found. Try changing the user, PSS search, or state."
    )
else:
    # Two compact cards per row
    cards_per_page = 20

    if "intimation_card_page" not in st.session_state:
        st.session_state.intimation_card_page = 0

    total_pages = max(
        1, (len(display_df) + cards_per_page - 1) // cards_per_page
    )

    # Reset the current page if filters reduce the result count.
    st.session_state.intimation_card_page = min(
        st.session_state.intimation_card_page,
        total_pages - 1,
    )

    page_col1, page_col2, page_col3 = st.columns([2, 1, 2])

    with page_col1:
        st.caption(
            f"Page {st.session_state.intimation_card_page + 1} of {total_pages}"
        )

    with page_col2:
        page_number = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=st.session_state.intimation_card_page,
            key="intimation_card_page_select",
            label_visibility="collapsed",
        )
        st.session_state.intimation_card_page = page_number - 1

    start = st.session_state.intimation_card_page * cards_per_page
    page_df = display_df.iloc[start:start + cards_per_page]

    card_rows = list(page_df.iterrows())

    for i in range(0, len(card_rows), 2):
        left, right = st.columns(2, gap="medium")

        with left:
            _, row = card_rows[i]
            st.markdown(make_card(row), unsafe_allow_html=True)

        if i + 1 < len(card_rows):
            with right:
                _, row = card_rows[i + 1]
                st.markdown(make_card(row), unsafe_allow_html=True)
