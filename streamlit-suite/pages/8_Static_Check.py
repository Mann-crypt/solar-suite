import math
from typing import Dict, List, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st
from pyproj import Geod

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError:
    Nominatim = None
    GeocoderTimedOut = GeocoderServiceError = Exception


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Coordinate Region Mapper",
    page_icon="📍",
    layout="wide",
)

GEOD = Geod(ellps="WGS84")

POINT_NAMES = [
    "Center",
    "Corner 1",
    "Corner 2",
    "Corner 3",
    "Corner 4",
]

DEFAULT_DATA = pd.DataFrame(
    {
        "Point": [
            "Center",
            "Corner 1",
            "Corner 2",
            "Corner 3",
            "Corner 4",
        ],
        "Latitude": [
            28.6250,
            28.6100,
            28.6100,
            28.6400,
            28.6400,
        ],
        "Longitude": [
            77.2150,
            77.2000,
            77.2300,
            77.2300,
            77.2000,
        ],
    }
)


# ============================================================
# GEOGRAPHIC CALCULATIONS
# ============================================================

def distance_km(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """
    Geodesic distance between two points.

    Input format:
        (latitude, longitude)
    """
    _, _, meters = GEOD.inv(
        a[1],
        a[0],
        b[1],
        b[0],
    )

    return abs(meters) / 1000.0


def polygon_area_km2(
    points: List[Tuple[float, float]]
) -> float:
    """
    Geodesic polygon area using WGS84.
    """

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    area_m2, _ = GEOD.polygon_area_perimeter(
        lons,
        lats,
    )

    return abs(area_m2) / 1_000_000.0


def polygon_perimeter_km(
    points: List[Tuple[float, float]]
) -> float:

    total = 0.0

    for i in range(len(points)):
        total += distance_km(
            points[i],
            points[(i + 1) % len(points)],
        )

    return total


def midpoint(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[float, float]:

    return (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
    )


def zoom_for_extent(
    points: List[Tuple[float, float]]
) -> float:

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)

    span = max(lat_span, lon_span)

    if span <= 0.001:
        return 15

    if span <= 0.003:
        return 14

    if span <= 0.01:
        return 13

    if span <= 0.03:
        return 12

    if span <= 0.08:
        return 11

    if span <= 0.2:
        return 9

    if span <= 0.5:
        return 8

    if span <= 1:
        return 7

    return 5


# ============================================================
# REVERSE GEOCODING
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def reverse_geocode(
    lat: float,
    lon: float,
) -> Dict[str, str]:

    if Nominatim is None:
        return {
            "status": "geopy is not installed."
        }

    try:

        geolocator = Nominatim(
            user_agent="streamlit-coordinate-region-mapper/1.0"
        )

        location = geolocator.reverse(
            (lat, lon),
            exactly_one=True,
            language="en",
            zoom=18,
        )

        if not location:
            return {
                "status": "No address found."
            }

        raw = location.raw.get(
            "address",
            {}
        )

        return {
            "road": raw.get("road", ""),

            "locality": (
                raw.get("city")
                or raw.get("town")
                or raw.get("village")
                or ""
            ),

            "district": (
                raw.get("city_district")
                or raw.get("county")
                or ""
            ),

            "state": raw.get(
                "state",
                ""
            ),

            "country": raw.get(
                "country",
                ""
            ),

            "postcode": raw.get(
                "postcode",
                ""
            ),

            "display_name": location.address,

            "status": "OK",
        }

    except (
        GeocoderTimedOut,
        GeocoderServiceError,
    ) as exc:

        return {
            "status": f"Geocoder unavailable: {exc}"
        }

    except Exception as exc:

        return {
            "status": f"Reverse geocoding failed: {exc}"
        }


# ============================================================
# PARSE EXCEL COPY/PASTE
# ============================================================

def parse_excel_text(text: str) -> pd.DataFrame:

    if not text.strip():
        raise ValueError(
            "The pasted data is empty."
        )

    # Excel normally copies data as TAB separated text.
    try:
        df = pd.read_csv(
            pd.io.common.StringIO(text),
            sep="\t",
        )
    except Exception:
        df = pd.read_csv(
            pd.io.common.StringIO(text),
            sep=r"\s+",
            engine="python",
        )

    # Normalize column names
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # Allow common Excel naming variations
    rename_map = {}

    for col in df.columns:

        if col in [
            "point",
            "name",
            "location",
            "type",
        ]:
            rename_map[col] = "Point"

        elif col in [
            "latitude",
            "lat",
        ]:
            rename_map[col] = "Latitude"

        elif col in [
            "longitude",
            "long",
            "lon",
        ]:
            rename_map[col] = "Longitude"

    df = df.rename(
        columns=rename_map
    )

    required = [
        "Point",
        "Latitude",
        "Longitude",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
            + ". Expected Point, Latitude, Longitude."
        )

    df = df[required].copy()

    df["Point"] = (
        df["Point"]
        .astype(str)
        .str.strip()
    )

    df["Latitude"] = pd.to_numeric(
        df["Latitude"],
        errors="coerce",
    )

    df["Longitude"] = pd.to_numeric(
        df["Longitude"],
        errors="coerce",
    )

    if df[
        ["Latitude", "Longitude"]
    ].isna().any().any():

        raise ValueError(
            "Latitude/Longitude contains invalid values."
        )

    if (
        (df["Latitude"] < -90)
        | (df["Latitude"] > 90)
    ).any():

        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if (
        (df["Longitude"] < -180)
        | (df["Longitude"] > 180)
    ).any():

        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    return df


# ============================================================
# VALIDATE POINT TABLE
# ============================================================

def validate_points(
    df: pd.DataFrame,
) -> Tuple[bool, str]:

    if len(df) != 5:

        return (
            False,
            "Exactly 5 points are required: "
            "Center + Corner 1 + Corner 2 + Corner 3 + Corner 4.",
        )

    expected = set(POINT_NAMES)

    actual = set(
        df["Point"].astype(str).str.strip()
    )

    missing = expected - actual
    extra = actual - expected

    if missing:

        return (
            False,
            "Missing point(s): "
            + ", ".join(sorted(missing)),
        )

    if extra:

        return (
            False,
            "Unknown point(s): "
            + ", ".join(sorted(extra)),
        )

    if df["Point"].duplicated().any():

        return (
            False,
            "Each point name must appear only once.",
        )

    return True, ""


# ============================================================
# BUILD MAP
# ============================================================

def build_map(
    points: Dict[str, Tuple[float, float]]
) -> pdk.Deck:

    corners = [
        points["Corner 1"],
        points["Corner 2"],
        points["Corner 3"],
        points["Corner 4"],
    ]

    center = points["Center"]

    # --------------------------------------------------------
    # Polygon
    # --------------------------------------------------------

    polygon_data = pd.DataFrame(
        [
            {
                "polygon": [
                    [lon, lat]
                    for lat, lon in (
                        corners
                        + [corners[0]]
                    )
                ],
                "name": "Selected Region",
            }
        ]
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    boundary_data = pd.DataFrame(
        [
            {
                "path": [
                    [lon, lat]
                    for lat, lon in (
                        corners
                        + [corners[0]]
                    )
                ],
                "name": "Region Boundary",
            }
        ]
    )

    # --------------------------------------------------------
    # Center → Corner Lines
    # --------------------------------------------------------

    center_lines = []

    for i, corner in enumerate(
        corners,
        start=1,
    ):

        center_lines.append(
            {
                "path": [
                    [center[1], center[0]],
                    [corner[1], corner[0]],
                ],
                "name": f"Center → C{i}",
            }
        )

    center_line_data = pd.DataFrame(
        center_lines
    )

    # --------------------------------------------------------
    # Markers
    # --------------------------------------------------------

    marker_rows = []

    for i in range(1, 5):

        lat, lon = points[
            f"Corner {i}"
        ]

        marker_rows.append(
            {
                "name": f"Corner {i}",
                "label": f"C{i}",
                "lat": lat,
                "lon": lon,
                "type": "Corner",
            }
        )

    marker_rows.append(
        {
            "name": "Center",
            "label": "CENTER",
            "lat": center[0],
            "lon": center[1],
            "type": "Center",
        }
    )

    markers = pd.DataFrame(
        marker_rows
    )

    # --------------------------------------------------------
    # Distance Labels
    # --------------------------------------------------------

    distance_labels = []

    # Boundary distances
    for i in range(4):

        a = corners[i]
        b = corners[
            (i + 1) % 4
        ]

        mid_lat, mid_lon = midpoint(
            a,
            b,
        )

        distance_labels.append(
            {
                "lat": mid_lat,
                "lon": mid_lon,
                "text": (
                    f"{distance_km(a, b):.2f} km"
                ),
            }
        )

    # Center distances
    for i, corner in enumerate(
        corners,
        start=1,
    ):

        mid_lat, mid_lon = midpoint(
            center,
            corner,
        )

        distance_labels.append(
            {
                "lat": mid_lat,
                "lon": mid_lon,
                "text": (
                    f"C→C{i}: "
                    f"{distance_km(center, corner):.2f} km"
                ),
            }
        )

    distance_label_data = pd.DataFrame(
        distance_labels
    )

    # --------------------------------------------------------
    # Map Center
    # --------------------------------------------------------

    all_points = corners + [center]

    map_center_lat = sum(
        p[0]
        for p in all_points
    ) / len(all_points)

    map_center_lon = sum(
        p[1]
        for p in all_points
    ) / len(all_points)

    # --------------------------------------------------------
    # Layers
    # --------------------------------------------------------

    layers = [

        # Polygon
        pdk.Layer(
            "PolygonLayer",
            data=polygon_data,
            get_polygon="polygon",
            get_fill_color=[
                50,
                120,
                220,
                55,
            ],
            get_line_color=[
                30,
                80,
                160,
                230,
            ],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        ),

        # Boundary
        pdk.Layer(
            "PathLayer",
            data=boundary_data,
            get_path="path",
            get_color=[
                30,
                80,
                160,
                230,
            ],
            width_min_pixels=3,
            pickable=True,
        ),

        # Center lines
        pdk.Layer(
            "PathLayer",
            data=center_line_data,
            get_path="path",
            get_color=[
                120,
                120,
                120,
                170,
            ],
            width_min_pixels=2,
        ),

        # Corner markers
        pdk.Layer(
            "ScatterplotLayer",
            data=markers[
                markers["type"] == "Corner"
            ],
            get_position="[lon, lat]",
            get_radius=90,
            radius_min_pixels=7,
            radius_max_pixels=16,
            get_fill_color=[
                30,
                90,
                180,
                230,
            ],
            get_line_color=[
                255,
                255,
                255,
                255,
            ],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        ),

        # Center marker
        pdk.Layer(
            "ScatterplotLayer",
            data=markers[
                markers["type"] == "Center"
            ],
            get_position="[lon, lat]",
            get_radius=130,
            radius_min_pixels=9,
            radius_max_pixels=20,
            get_fill_color=[
                220,
                70,
                50,
                240,
            ],
            get_line_color=[
                255,
                255,
                255,
                255,
            ],
            line_width_min_pixels=2,
            pickable=True,
        ),

        # Point labels
        pdk.Layer(
            "TextLayer",
            data=markers,
            get_position="[lon, lat]",
            get_text="label",
            get_size=15,
            get_color=[
                20,
                20,
                20,
                255,
            ],
            get_pixel_offset=[
                0,
                -18,
            ],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
        ),

        # Distance labels
        pdk.Layer(
            "TextLayer",
            data=distance_label_data,
            get_position="[lon, lat]",
            get_text="text",
            get_size=11,
            get_color=[
                50,
                50,
                50,
                255,
            ],
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
        ),
    ]

    return pdk.Deck(

        layers=layers,

        initial_view_state=pdk.ViewState(
            latitude=map_center_lat,
            longitude=map_center_lon,
            zoom=zoom_for_extent(
                all_points
            ),
            pitch=0,
            bearing=0,
        ),

        map_style=None,

        tooltip={
            "html": (
                "<b>{name}</b>"
                "<br/>Latitude: {lat}"
                "<br/>Longitude: {lon}"
            ),
            "style": {
                "backgroundColor": (
                    "rgba(30,30,30,0.9)"
                ),
                "color": "white",
            },
        },
    )


# ============================================================
# UI
# ============================================================

st.title(
    "📍 Coordinate Region Mapper"
)

st.caption(
    "Paste coordinates from Excel or enter them manually."
)


# ============================================================
# EXCEL PASTE SECTION
# ============================================================

st.subheader(
    "1. Paste Coordinates from Excel"
)

st.markdown(
    "Copy the **Point, Latitude and Longitude** columns from Excel and paste them below."
)

example_text = (
    "Point\tLatitude\tLongitude\n"
    "Center\t28.6250\t77.2150\n"
    "Corner 1\t28.6100\t77.2000\n"
    "Corner 2\t28.6100\t77.2300\n"
    "Corner 3\t28.6400\t77.2300\n"
    "Corner 4\t28.6400\t77.2000"
)

paste_text = st.text_area(
    "Excel data",
    value="",
    placeholder=example_text,
    height=150,
    help=(
        "Paste exactly three columns: "
        "Point, Latitude, Longitude."
    ),
)

p1, p2 = st.columns(
    [1, 1]
)

with p1:

    parse_button = st.button(
        "📋 Load Pasted Data",
        use_container_width=True,
    )

with p2:

    reset_button = st.button(
        "↩ Reset to Example",
        use_container_width=True,
    )


if reset_button:

    st.session_state[
        "coordinate_table"
    ] = DEFAULT_DATA.copy()

    st.rerun()


if parse_button:

    try:

        parsed = parse_excel_text(
            paste_text
        )

        valid, message = validate_points(
            parsed
        )

        if not valid:

            st.error(message)

        else:

            st.session_state[
                "coordinate_table"
            ] = parsed

            st.success(
                "Coordinates loaded successfully."
            )

    except Exception as exc:

        st.error(
            f"Could not read the pasted Excel data: {exc}"
        )


# ============================================================
# TABLE SECTION
# ============================================================

st.divider()

st.subheader(
    "2. Coordinate Table"
)

if "coordinate_table" not in st.session_state:

    st.session_state[
        "coordinate_table"
    ] = DEFAULT_DATA.copy()


edited_df = st.data_editor(
    st.session_state[
        "coordinate_table"
    ],
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={

        "Point": st.column_config.TextColumn(
            "Point",
            disabled=True,
        ),

        "Latitude": st.column_config.NumberColumn(
            "Latitude",
            format="%.6f",
            min_value=-90.0,
            max_value=90.0,
            step=0.000001,
        ),

        "Longitude": st.column_config.NumberColumn(
            "Longitude",
            format="%.6f",
            min_value=-180.0,
            max_value=180.0,
            step=0.000001,
        ),
    },
)

st.session_state[
    "coordinate_table"
] = edited_df


# ============================================================
# GENERATE
# ============================================================

generate = st.button(
    "🚀 Generate Region",
    type="primary",
    use_container_width=True,
)


if generate:

    df = st.session_state[
        "coordinate_table"
    ].copy()

    valid, message = validate_points(
        df
    )

    if not valid:

        st.error(message)

        st.stop()

    points = {}

    for _, row in df.iterrows():

        points[
            row["Point"]
        ] = (
            float(row["Latitude"]),
            float(row["Longitude"]),
        )

    st.session_state[
        "region_points"
    ] = points


# ============================================================
# MAP
# ============================================================

if "region_points" in st.session_state:

    points = st.session_state[
        "region_points"
    ]

    corners = [
        points["Corner 1"],
        points["Corner 2"],
        points["Corner 3"],
        points["Corner 4"],
    ]

    center = points["Center"]

    st.divider()

    st.subheader(
        "3. Interactive Region Map"
    )

    try:

        deck = build_map(
            points
        )

        st.pydeck_chart(
            deck,
            width="stretch",
            height=620,
        )

    except Exception as exc:

        st.error(
            f"Unable to render map: {exc}"
        )


    # ========================================================
    # CALCULATIONS
    # ========================================================

    boundary_rows = []

    for i in range(4):

        start = corners[i]
        end = corners[
            (i + 1) % 4
        ]

        boundary_rows.append(
            {
                "Segment": (
                    f"C{i + 1} → "
                    f"C{(i + 1) % 4 + 1}"
                ),
                "Distance (km)": (
                    distance_km(
                        start,
                        end,
                    )
                ),
            }
        )

    boundary_df = pd.DataFrame(
        boundary_rows
    )

    center_rows = []

    for i, corner in enumerate(
        corners,
        start=1,
    ):

        center_rows.append(
            {
                "Segment": (
                    f"Center → C{i}"
                ),
                "Distance (km)": (
                    distance_km(
                        center,
                        corner,
                    )
                ),
            }
        )

    center_df = pd.DataFrame(
        center_rows
    )

    area = polygon_area_km2(
        corners
    )

    perimeter = polygon_perimeter_km(
        corners
    )

    average_center_distance = (
        center_df[
            "Distance (km)"
        ].mean()
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "4. Region Summary"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Area",
            f"{area:,.4f} km²",
        )

    with m2:
        st.metric(
            "Perimeter",
            f"{perimeter:,.3f} km",
        )

    with m3:
        st.metric(
            "Avg. Center Distance",
            f"{average_center_distance:,.3f} km",
        )

    with m4:
        st.metric(
            "Center",
            (
                f"{center[0]:.5f}, "
                f"{center[1]:.5f}"
            ),
        )


    # ========================================================
    # DISTANCE TABLES
    # ========================================================

    st.subheader(
        "5. Distance Analysis"
    )

    tab1, tab2 = st.tabs(
        [
            "Boundary Distances",
            "Center Distances",
        ]
    )

    with tab1:

        st.dataframe(
            boundary_df.style.format(
                {
                    "Distance (km)": "{:.3f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            f"**Total perimeter:** "
            f"{perimeter:.3f} km"
        )

    with tab2:

        st.dataframe(
            center_df.style.format(
                {
                    "Distance (km)": "{:.3f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # DISTRICT INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "6. Center Point Location"
    )

    st.caption(
        "Administrative information is obtained "
        "using the center coordinate."
    )

    with st.spinner(
        "Finding district and administrative location..."
    ):

        geo = reverse_geocode(
            center[0],
            center[1],
        )

    if geo.get(
        "status"
    ) == "OK":

        g1, g2, g3, g4 = st.columns(4)

        with g1:

            st.markdown(
                "**Country**"
            )

            st.write(
                geo.get(
                    "country"
                )
                or "Not available"
            )

        with g2:

            st.markdown(
                "**State**"
            )

            st.write(
                geo.get(
                    "state"
                )
                or "Not available"
            )

        with g3:

            st.markdown(
                "**District**"
            )

            st.write(
                geo.get(
                    "district"
                )
                or "Not available"
            )

        with g4:

            st.markdown(
                "**City / Locality**"
            )

            st.write(
                geo.get(
                    "locality"
                )
                or "Not available"
            )

        if geo.get(
            "postcode"
        ):

            st.write(
                f"**PIN / Postcode:** "
                f"{geo['postcode']}"
            )

        if geo.get(
            "road"
        ):

            st.write(
                f"**Road / Area:** "
                f"{geo['road']}"
            )

        if geo.get(
            "display_name"
        ):

            st.caption(
                geo["display_name"]
            )

    else:

        st.warning(
            "Could not retrieve "
            "administrative information. "
            + geo.get(
                "status",
                "Unknown error",
            )
        )


    # ========================================================
    # EXPORT
    # ========================================================

    st.divider()

    st.subheader(
        "7. Export Results"
    )

    result_rows = []

    for i, corner in enumerate(
        corners,
        start=1,
    ):

        result_rows.append(
            {
                "Point": f"C{i}",
                "Latitude": corner[0],
                "Longitude": corner[1],
                "Distance_from_Center_km": (
                    distance_km(
                        center,
                        corner,
                    )
                ),
            }
        )

    result_rows.append(
        {
            "Point": "Center",
            "Latitude": center[0],
            "Longitude": center[1],
            "Distance_from_Center_km": 0.0,
        }
    )

    result_df = pd.DataFrame(
        result_rows
    )

    csv = (
        result_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇ Download Coordinates + Distances CSV",
        data=csv,
        file_name=(
            "coordinate_region_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )
