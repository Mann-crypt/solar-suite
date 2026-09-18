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
# Coordinate Region Mapper
# Streamlit + PyDeck + PyProj + Geopy
# ============================================================

st.set_page_config(
    page_title="Coordinate Region Mapper",
    page_icon="📍",
    layout="wide",
)

# WGS84 ellipsoid for accurate geodesic distance/area calculations.
GEOD = Geod(ellps="WGS84")

POINT_NAMES = ["Corner 1", "Corner 2", "Corner 3", "Corner 4", "Center"]
DEFAULTS = {
    "Corner 1": (28.6100, 77.2000),
    "Corner 2": (28.6100, 77.2300),
    "Corner 3": (28.6400, 77.2300),
    "Corner 4": (28.6400, 77.2000),
    "Center": (28.6250, 77.2150),
}


def valid_lat(lat: float) -> bool:
    return -90 <= lat <= 90


def valid_lon(lon: float) -> bool:
    return -180 <= lon <= 180


def distance_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Geodesic distance between (lat, lon) pairs."""
    _, _, meters = GEOD.inv(a[1], a[0], b[1], b[0])
    return abs(meters) / 1000.0


def polygon_area_km2(points: List[Tuple[float, float]]) -> float:
    """Geodesic polygon area on WGS84."""
    lons = [p[1] for p in points]
    lats = [p[0] for p in points]
    area_m2, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1_000_000.0


def polygon_perimeter_km(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(points)):
        total += distance_km(points[i], points[(i + 1) % len(points)])
    return total


def midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Simple coordinate midpoint, suitable for short regions."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def zoom_for_extent(points: List[Tuple[float, float]]) -> float:
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


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> Dict[str, str]:
    """Reverse geocode the center point using OpenStreetMap Nominatim."""
    if Nominatim is None:
        return {"status": "geopy is not installed"}

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
            return {"status": "No address found"}

        raw = location.raw.get("address", {})

        return {
            "road": raw.get("road", ""),
            "locality": raw.get("city") or raw.get("town") or raw.get("village") or "",
            "district": raw.get("city_district") or raw.get("county") or "",
            "state": raw.get("state", ""),
            "country": raw.get("country", ""),
            "postcode": raw.get("postcode", ""),
            "display_name": location.address,
            "status": "OK",
        }

    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        return {"status": f"Geocoder unavailable: {exc}"}
    except Exception as exc:
        return {"status": f"Reverse geocoding failed: {exc}"}


def build_map(points: Dict[str, Tuple[float, float]]) -> pdk.Deck:
    corners = [points[f"Corner {i}"] for i in range(1, 5)]
    center = points["Center"]

    polygon_data = pd.DataFrame(
        [
            {
                "polygon": [[lon, lat] for lat, lon in corners + [corners[0]]],
                "name": "Selected Region",
            }
        ]
    )

    # Boundary path
    path_data = pd.DataFrame(
        [
            {
                "path": [[lon, lat] for lat, lon in corners + [corners[0]]],
                "name": "Region Boundary",
            }
        ]
    )

    # Center-to-corner lines
    center_lines = []
    for i, corner in enumerate(corners, start=1):
        center_lines.append(
            {
                "path": [
                    [center[1], center[0]],
                    [corner[1], corner[0]],
                ],
                "name": f"Center → C{i}",
            }
        )

    center_line_data = pd.DataFrame(center_lines)

    marker_rows = []
    for i in range(1, 5):
        lat, lon = points[f"Corner {i}"]
        marker_rows.append(
            {
                "name": f"C{i}",
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

    markers = pd.DataFrame(marker_rows)

    # Distance labels on each boundary edge
    distance_labels = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        mid_lat, mid_lon = midpoint(a, b)
        distance_labels.append(
            {
                "lat": mid_lat,
                "lon": mid_lon,
                "text": f"{distance_km(a, b):.2f} km",
            }
        )

    # Distance labels from center to corners
    for i, corner in enumerate(corners, start=1):
        mid_lat, mid_lon = midpoint(center, corner)
        distance_labels.append(
            {
                "lat": mid_lat,
                "lon": mid_lon,
                "text": f"C→C{i}: {distance_km(center, corner):.2f} km",
            }
        )

    distance_label_data = pd.DataFrame(distance_labels)

    all_points = corners + [center]
    map_center_lat = sum(p[0] for p in all_points) / len(all_points)
    map_center_lon = sum(p[1] for p in all_points) / len(all_points)

    layers = [
        pdk.Layer(
            "PolygonLayer",
            data=polygon_data,
            get_polygon="polygon",
            get_fill_color=[50, 120, 220, 55],
            get_line_color=[30, 80, 160, 220],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color=[30, 80, 160, 230],
            width_min_pixels=3,
            pickable=True,
        ),
        pdk.Layer(
            "PathLayer",
            data=center_line_data,
            get_path="path",
            get_color=[120, 120, 120, 170],
            width_min_pixels=2,
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=markers[markers["type"] == "Corner"],
            get_position="[lon, lat]",
            get_radius=90,
            radius_min_pixels=7,
            radius_max_pixels=16,
            get_fill_color=[30, 90, 180, 230],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=markers[markers["type"] == "Center"],
            get_position="[lon, lat]",
            get_radius=130,
            radius_min_pixels=9,
            radius_max_pixels=20,
            get_fill_color=[220, 70, 50, 240],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=markers,
            get_position="[lon, lat]",
            get_text="label",
            get_size=15,
            get_color=[20, 20, 20, 255],
            get_pixel_offset=[0, -18],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            data=distance_label_data,
            get_position="[lon, lat]",
            get_text="text",
            get_size=11,
            get_color=[50, 50, 50, 255],
            get_pixel_offset=[0, 0],
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            pickable=False,
        ),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=map_center_lat,
            longitude=map_center_lon,
            zoom=zoom_for_extent(all_points),
            pitch=0,
            bearing=0,
        ),
        map_style=None,
        tooltip={
            "html": "<b>{name}</b><br/>Latitude: {lat}<br/>Longitude: {lon}",
            "style": {
                "backgroundColor": "rgba(30,30,30,0.9)",
                "color": "white",
            },
        },
    )


# ============================================================
# UI
# ============================================================

st.title("📍 Coordinate Region Mapper")
st.caption(
    "Enter four corner coordinates and one center coordinate to build a geographic region."
)

left, right = st.columns([0.82, 1.55], gap="large")

with left:
    st.subheader("Coordinate Input")

    point_values = {}

    for name in POINT_NAMES:
        default_lat, default_lon = DEFAULTS[name]

        st.markdown(f"**{name}**")

        c1, c2 = st.columns(2)

        with c1:
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(default_lat),
                step=0.0001,
                format="%.6f",
                key=f"{name}_lat",
            )

        with c2:
            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(default_lon),
                step=0.0001,
                format="%.6f",
                key=f"{name}_lon",
            )

        point_values[name] = (lat, lon)

    st.divider()

    generate = st.button(
        "Generate Region",
        type="primary",
        use_container_width=True,
    )

    if generate:
        st.session_state["region_points"] = point_values

with right:
    st.subheader("Interactive Map")

    if "region_points" not in st.session_state:
        st.info("Enter coordinates and click **Generate Region**.")
    else:
        points = st.session_state["region_points"]

        try:
            deck = build_map(points)
            st.pydeck_chart(deck, width="stretch", height=620)
        except Exception as exc:
            st.error(f"Unable to render the map: {exc}")


# ============================================================
# Calculations
# ============================================================

if "region_points" in st.session_state:
    points = st.session_state["region_points"]
    corners = [points[f"Corner {i}"] for i in range(1, 5)]
    center = points["Center"]

    # Boundary distances
    boundary_rows = []
    for i in range(4):
        start_name = f"C{i + 1}"
        end_name = f"C{(i + 1) % 4 + 1}"
        start = corners[i]
        end = corners[(i + 1) % 4]

        boundary_rows.append(
            {
                "Segment": f"{start_name} → {end_name}",
                "Distance (km)": distance_km(start, end),
            }
        )

    boundary_df = pd.DataFrame(boundary_rows)

    # Center distances
    center_rows = []
    for i, corner in enumerate(corners, start=1):
        center_rows.append(
            {
                "Segment": f"Center → C{i}",
                "Distance (km)": distance_km(center, corner),
            }
        )

    center_df = pd.DataFrame(center_rows)

    area = polygon_area_km2(corners)
    perimeter = polygon_perimeter_km(corners)
    avg_center_distance = center_df["Distance (km)"].mean()

    st.divider()
    st.subheader("Region Summary")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Area", f"{area:,.4f} km²")
    m2.metric("Perimeter", f"{perimeter:,.3f} km")
    m3.metric("Avg. Center Distance", f"{avg_center_distance:,.3f} km")
    m4.metric("Center", f"{center[0]:.5f}, {center[1]:.5f}")

    tab1, tab2, tab3 = st.tabs(
        ["Boundary Distances", "Center Distances", "Coordinates"]
    )

    with tab1:
        st.dataframe(
            boundary_df.style.format({"Distance (km)": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.write(f"**Total perimeter:** {perimeter:.3f} km")

    with tab2:
        st.dataframe(
            center_df.style.format({"Distance (km)": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        coordinate_rows = []

        for name, (lat, lon) in points.items():
            coordinate_rows.append(
                {
                    "Point": name,
                    "Latitude": lat,
                    "Longitude": lon,
                }
            )

        coordinates_df = pd.DataFrame(coordinate_rows)

        st.dataframe(
            coordinates_df.style.format(
                {
                    "Latitude": "{:.6f}",
                    "Longitude": "{:.6f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # District / Reverse Geocoding
    # ========================================================

    st.divider()
    st.subheader("📌 Center Point Location")

    st.caption(
        "Administrative information is obtained by reverse geocoding the center point."
    )

    with st.spinner("Finding district and administrative location..."):
        geo = reverse_geocode(center[0], center[1])

    if geo.get("status") == "OK":
        g1, g2, g3, g4 = st.columns(4)

        with g1:
            st.markdown("**Country**")
            st.write(geo.get("country") or "Not available")

        with g2:
            st.markdown("**State**")
            st.write(geo.get("state") or "Not available")

        with g3:
            st.markdown("**District**")
            st.write(geo.get("district") or "Not available")

        with g4:
            st.markdown("**City / Locality**")
            st.write(geo.get("locality") or "Not available")

        if geo.get("postcode"):
            st.write(f"**PIN / Postcode:** {geo['postcode']}")

        if geo.get("road"):
            st.write(f"**Road / Area:** {geo['road']}")

        if geo.get("display_name"):
            st.caption(geo["display_name"])

    else:
        st.warning(
            f"Could not retrieve administrative information. "
            f"{geo.get('status', 'Unknown error')}"
        )

    # ========================================================
    # Downloadable results
    # ========================================================

    result_rows = []

    for i, corner in enumerate(corners, start=1):
        result_rows.append(
            {
                "Point": f"C{i}",
                "Latitude": corner[0],
                "Longitude": corner[1],
                "Distance_from_Center_km": distance_km(center, corner),
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

    result_df = pd.DataFrame(result_rows)

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Coordinates + Distances CSV",
        data=csv,
        file_name="coordinate_region_results.csv",
        mime="text/csv",
    )
