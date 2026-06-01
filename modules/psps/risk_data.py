"""modules/psps/risk_data.py — Synthetic wildfire risk layers for the IEEE 9500-Node feeder.

Geographic context: Kennewick / Richland / Pasco (Tri-Cities), WA.
  Feeder extent:  lon -119.27 → -119.07 W,  lat 46.60 → 46.77 N
  Source bus:     ~(-119.07, 46.70)  →  near Richland / Hanford Site (northeast)

Risk data is generated procedurally at import time using geographic scoring:
  - Distance from the urban core (Richland/Kennewick, NE quadrant)
  - Proximity to the Columbia River corridor (low risk: agriculture/riparian)
  - Western hills / Horse Heaven Hills direction (higher risk: remote shrub-steppe)
  - Calibrated Gaussian noise for realistic patchwork variation

Grid sizes:
  Risk   : 10 × 10 = 100 polygons  (seed 7)
  Fuel   : 10 × 10 = 100 polygons  (seed 13)
  Total on screen: up to 200 (risk + fuel layers)

All polygons use a shared jittered-vertex grid so edges are shared between
adjacent cells — no gaps, no overlaps, realistic irregular boundaries.

All coordinates: X = longitude (negative, ~-119), Y = latitude (~46.6–46.77).
"""

from __future__ import annotations
import math
import random

# ── Feeder extent ──────────────────────────────────────────────────────────────

_LON_MIN, _LON_MAX = -119.27, -119.07
_LAT_MIN, _LAT_MAX =   46.60,   46.77

# ── Geographic anchors ─────────────────────────────────────────────────────────

_URBAN_CENTER  = (-119.095, 46.728)  # Richland / Kennewick urban core
_RIVER_LAT     =  46.645             # Columbia River approximate latitude
_RIVER_WIDTH   =  0.022              # Half-width of strong river influence (deg)

# Wildfire risk hotspots — concentrated in peripheral branch areas so
# the optimizer can isolate them by cutting individual laterals rather
# than the main feeder trunk.
_HOTSPOT_NW = (-119.252, 46.757)   # NW sagebrush ridgeline / exposed slopes
_HOTSPOT_SW = (-119.258, 46.612)   # SW dryland corridor / Horse Heaven Hills — far corner

# ── Grid dimensions ────────────────────────────────────────────────────────────

_N_RISK_LON, _N_RISK_LAT = 10, 10   # → 100 risk polygons
_N_FUEL_LON, _N_FUEL_LAT = 10, 10   # → 100 fuel polygons

# ── Tier colour palette ────────────────────────────────────────────────────────

TIER_META = {
    0: {"label": "No risk",       "color": "#78909C", "fill": "rgba(120,144,156,0.10)"},
    1: {"label": "Tier 1 — Low",  "color": "#F9A825", "fill": "rgba(249,168,37,0.20)"},
    2: {"label": "Tier 2 — Mod.", "color": "#E65100", "fill": "rgba(230,81,0,0.25)"},
    3: {"label": "Tier 3 — High", "color": "#B71C1C", "fill": "rgba(183,28,28,0.30)"},
}

FUEL_META = {
    "Urban/Developed":          {"color": "#9E9E9E", "fill": "rgba(158,158,158,0.25)"},
    "Wildland-Urban Interface": {"color": "#FF8F00", "fill": "rgba(255,143,0,0.22)"},
    "Irrigated Agriculture":    {"color": "#66BB6A", "fill": "rgba(102,187,106,0.25)"},
    "Dryland Cropland":         {"color": "#A1887F", "fill": "rgba(161,136,127,0.25)"},
    "Shrub-Steppe":             {"color": "#D4A017", "fill": "rgba(212,160,23,0.28)"},
    "Dry Grassland":            {"color": "#FDD835", "fill": "rgba(253,216,53,0.25)"},
    "Riparian":                 {"color": "#29B6F6", "fill": "rgba(41,182,246,0.20)"},
}

# ── Geographic scoring ─────────────────────────────────────────────────────────

def _risk_base_score(lon: float, lat: float) -> float:
    """Geographic risk score 0.0 (safe) → 1.0 (high risk).

    Uses two Gaussian hotspots in the peripheral branch areas (NW ridge,
    SW dryland corridor) plus a gentle background gradient from the urban
    core.  Hotspots are placed at the tips of lateral branches so the
    optimizer can isolate them by opening branch switches rather than the
    main feeder trunk.
    """
    # NW hotspot (exposed sagebrush ridgeline)
    d_nw = math.sqrt(
        ((lon - _HOTSPOT_NW[0]) / 0.040) ** 2 +
        ((lat - _HOTSPOT_NW[1]) / 0.028) ** 2
    )
    h_nw = math.exp(-0.5 * d_nw ** 2)

    # SW hotspot (Horse Heaven Hills dryland approach — far SW corner only)
    d_sw = math.sqrt(
        ((lon - _HOTSPOT_SW[0]) / 0.028) ** 2 +
        ((lat - _HOTSPOT_SW[1]) / 0.018) ** 2
    )
    h_sw = math.exp(-0.5 * d_sw ** 2)

    hotspot_score = max(h_nw, h_sw)

    # Background: gentle gradient away from urban / safe core.
    # Capped at 0.48 — below the Tier 3 threshold — so background alone
    # never reaches Tier 3.  Only the hotspot Gaussian can do that.
    d_urban = math.sqrt(
        ((lon - _URBAN_CENTER[0]) / 0.12) ** 2 +
        ((lat - _URBAN_CENTER[1]) / 0.10) ** 2
    )
    background = min(0.48, d_urban * 0.43)

    score = max(hotspot_score, background)

    # River corridor suppresses risk (irrigated / riparian buffer)
    river_prox = max(0.0, 1.0 - abs(lat - _RIVER_LAT) / _RIVER_WIDTH)
    score *= 1.0 - 0.55 * river_prox

    return score


def _fuel_base_type(lon: float, lat: float, rng: random.Random) -> str:
    """Return a realistic fuel type for a cell centre, with controlled noise."""
    d_urban = math.sqrt(
        ((lon - _URBAN_CENTER[0]) / 0.08) ** 2 +
        ((lat - _URBAN_CENTER[1]) / 0.06) ** 2
    )
    river_prox = max(0.0, 1.0 - abs(lat - _RIVER_LAT) / 0.018)

    # River band: riparian / irrigated
    if river_prox > 0.55:
        return "Riparian" if rng.random() > 0.45 else "Irrigated Agriculture"

    # Urban core
    if d_urban < 0.38:
        return "Urban/Developed"

    # Wildland-Urban Interface transition ring
    if d_urban < 0.72:
        return "Wildland-Urban Interface"

    # NE quadrant: irrigated / dryland mixed
    if lat > 46.67 and lon > -119.175:
        return "Irrigated Agriculture" if rng.random() > 0.35 else "Dryland Cropland"

    # Far west / south: shrub-steppe / grassland
    if lon < -119.19 or lat < 46.635:
        return "Shrub-Steppe" if lat > 46.62 else "Dry Grassland"

    # Mid transition
    return "Dryland Cropland" if rng.random() > 0.45 else "Shrub-Steppe"

# ── Jittered-vertex grid builder ───────────────────────────────────────────────

def _make_vertices(n_lon: int, n_lat: int, jitter: float, rng: random.Random) -> dict:
    """Build an (n_lon+1)×(n_lat+1) jittered vertex grid.

    Boundary vertices are kept fixed so polygons don't protrude outside the
    feeder extent. Interior vertices are perturbed by up to ±jitter × cell_size,
    producing realistically irregular cell boundaries.
    """
    cell_w = (_LON_MAX - _LON_MIN) / n_lon
    cell_h = (_LAT_MAX - _LAT_MIN) / n_lat
    verts: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(n_lon + 1):
        for j in range(n_lat + 1):
            base_lon = _LON_MIN + i * cell_w
            base_lat = _LAT_MIN + j * cell_h
            if 0 < i < n_lon and 0 < j < n_lat:
                dlon = rng.uniform(-jitter * cell_w, jitter * cell_w)
                dlat = rng.uniform(-jitter * cell_h, jitter * cell_h)
            else:
                dlon = dlat = 0.0
            verts[(i, j)] = (base_lon + dlon, base_lat + dlat)
    return verts


def _cell_polygon(
    verts: dict[tuple[int, int], tuple[float, float]], i: int, j: int
) -> tuple[list[float], list[float]]:
    sw = verts[(i,     j    )]
    se = verts[(i + 1, j    )]
    ne = verts[(i + 1, j + 1)]
    nw = verts[(i,     j + 1)]
    return (
        [sw[0], se[0], ne[0], nw[0], sw[0]],
        [sw[1], se[1], ne[1], nw[1], sw[1]],
    )

# ── Name / description helpers ─────────────────────────────────────────────────

_LON_LABELS = [
    (-119.27, -119.21, "Far-Western"),
    (-119.21, -119.17, "Western"),
    (-119.17, -119.13, "Central-West"),
    (-119.13, -119.09, "Central-East"),
    (-119.09, -119.07, "Eastern"),
]
_LAT_LABELS = [
    (46.60, 46.64, "Southern"),
    (46.64, 46.68, "River-Corridor"),
    (46.68, 46.72, "Central"),
    (46.72, 46.77, "Northern"),
]

_TIER_SUFFIXES = {
    3: ["Sagebrush Flats", "Remote Shrubland", "Canyon Approach", "Ridge Exposure",
        "Dry Hillside", "Upland Steppe", "Horse Heaven Spur", "Bunchgrass Bench",
        "Windswept Slope", "Exposed Ridgeline"],
    2: ["Transition Zone", "Semi-Rural Margin", "Dryland Interface", "Rangeland Strip",
        "Grassland Buffer", "Rural Scatter", "Wind Corridor", "Post-Harvest Zone",
        "Outskirt Fringe", "Open Shrubland"],
    1: ["Urban Fringe", "Irrigated Corridor", "Residential Block", "Agricultural Patch",
        "Developed Area", "Low-Fuel Zone", "Landscaped Margin", "Riverine Flat",
        "Suburban Core", "Orchard/Turf Block"],
}

def _zone_name(tier: int, lon: float, lat: float, idx: int) -> str:
    lon_lab = next(
        (lb for lo, hi, lb in _LON_LABELS if lo <= lon < hi),
        "Western",
    )
    lat_lab = next(
        (lb for lo, hi, lb in _LAT_LABELS if lo <= lat < hi),
        "Central",
    )
    suffixes = _TIER_SUFFIXES[tier]
    suffix = suffixes[idx % len(suffixes)]
    return f"{lat_lab} {lon_lab} — {suffix}"


_FUEL_DESCS = {
    "Urban/Developed":
        "Impervious surfaces, irrigated turf and landscaping. Very low fire spread potential.",
    "Wildland-Urban Interface":
        "Transition between developed land and native vegetation. Structures at risk; "
        "moderate to high fuel loading depending on parcel management.",
    "Irrigated Agriculture":
        "Row crops and orchards under active irrigation. Low fuel load when green; "
        "moderate risk in post-harvest dry stubble season.",
    "Dryland Cropland":
        "Non-irrigated wheat and barley. Highly flammable standing grain in summer; "
        "fast-spreading stubble fires post-harvest.",
    "Shrub-Steppe":
        "Big sagebrush / bluebunch wheatgrass. Primary vegetation of the Columbia Plateau. "
        "Burns intensely under low-humidity Palouser (easterly) wind events.",
    "Dry Grassland":
        "Cheatgrass-dominated annual grassland. Flashy, fast-spreading fuel; "
        "peak risk late spring through mid-summer.",
    "Riparian":
        "Columbia River banks, wetlands, and irrigated bottomland. Naturally moist; "
        "acts as a fire break under most conditions.",
}

_TIER_DESCS = {
    3: "High ignition probability. Remote location, dense native fuel, limited suppression access.",
    2: "Moderate risk transitional area. Mixed land use, variable fuel continuity, "
       "some suppression access.",
    1: "Lower risk zone. Irrigated or developed land cover, good hydrant and road coverage.",
}

# ── Polygon generators ─────────────────────────────────────────────────────────

def _generate_risk_polygons() -> list[dict]:
    rng   = random.Random(7)
    verts = _make_vertices(_N_RISK_LON, _N_RISK_LAT, 0.22, rng)
    cell_w = (_LON_MAX - _LON_MIN) / _N_RISK_LON
    cell_h = (_LAT_MAX - _LAT_MIN) / _N_RISK_LAT

    out = []
    idx_by_tier: dict[int, int] = {1: 0, 2: 0, 3: 0}

    for i in range(_N_RISK_LON):
        for j in range(_N_RISK_LAT):
            cx = _LON_MIN + (i + 0.5) * cell_w
            cy = _LAT_MIN + (j + 0.5) * cell_h

            base  = _risk_base_score(cx, cy)
            noise = rng.gauss(0, 0.04)
            score = max(0.0, min(1.0, base + noise))

            tier = 1 if score < 0.28 else (2 if score < 0.64 else 3)

            lons, lats = _cell_polygon(verts, i, j)
            name = _zone_name(tier, cx, cy, idx_by_tier[tier])
            idx_by_tier[tier] += 1

            out.append({
                "tier": tier,
                "name": name,
                "desc": _TIER_DESCS[tier],
                "lons": lons,
                "lats": lats,
                "grid": (i, j),
            })

    return out


def _generate_fuel_polygons() -> list[dict]:
    rng   = random.Random(13)
    verts = _make_vertices(_N_FUEL_LON, _N_FUEL_LAT, 0.20, rng)
    cell_w = (_LON_MAX - _LON_MIN) / _N_FUEL_LON
    cell_h = (_LAT_MAX - _LAT_MIN) / _N_FUEL_LAT

    out = []
    for i in range(_N_FUEL_LON):
        for j in range(_N_FUEL_LAT):
            cx = _LON_MIN + (i + 0.5) * cell_w
            cy = _LAT_MIN + (j + 0.5) * cell_h

            ftype = _fuel_base_type(cx, cy, rng)
            lons, lats = _cell_polygon(verts, i, j)

            out.append({
                "type": ftype,
                "desc": _FUEL_DESCS[ftype],
                "lons": lons,
                "lats": lats,
                "grid": (i, j),
            })

    return out

# ── Module-level generated data ────────────────────────────────────────────────

_RISK_POLYGONS: list[dict] = _generate_risk_polygons()
_FUEL_POLYGONS: list[dict] = _generate_fuel_polygons()

# Fast O(1) grid-index lookup maps: (i, j) → polygon
_RISK_GRID: dict[tuple[int, int], dict] = {p["grid"]: p for p in _RISK_POLYGONS}
_FUEL_GRID: dict[tuple[int, int], dict] = {p["grid"]: p for p in _FUEL_POLYGONS}

# ── Point-in-polygon (ray casting) ────────────────────────────────────────────

def _pip(lon: float, lat: float, lons: list[float], lats: list[float]) -> bool:
    inside = False
    n = len(lons) - 1          # closed polygon: last vertex == first
    j = n - 1
    for i in range(n):
        xi, yi = lons[i], lats[i]
        xj, yj = lons[j], lats[j]
        if (yi > lat) != (yj > lat):
            if lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _lookup(
    lon: float, lat: float,
    n_lon: int, n_lat: int,
    grid: dict[tuple[int, int], dict],
) -> dict | None:
    """Grid-index lookup with single-ring neighbour fallback for jitter edges."""
    fi = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN) * n_lon
    fj = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN) * n_lat
    ci = max(0, min(n_lon - 1, int(fi)))
    cj = max(0, min(n_lat - 1, int(fj)))

    # Primary cell — almost always correct
    p = grid.get((ci, cj))
    if p and _pip(lon, lat, p["lons"], p["lats"]):
        return p

    # Neighbour ring — handles the rare jitter boundary case
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            key = (ci + di, cj + dj)
            p = grid.get(key)
            if p and _pip(lon, lat, p["lons"], p["lats"]):
                return p

    return None

# ── Public API ─────────────────────────────────────────────────────────────────

def get_risk_tier(lon: float, lat: float) -> int:
    p = _lookup(lon, lat, _N_RISK_LON, _N_RISK_LAT, _RISK_GRID)
    return p["tier"] if p else 0


def get_fuel_type(lon: float, lat: float) -> str:
    p = _lookup(lon, lat, _N_FUEL_LON, _N_FUEL_LAT, _FUEL_GRID)
    return p["type"] if p else "Shrub-Steppe"


def get_risk_zones_for_plotly() -> list[dict]:
    out = []
    for p in _RISK_POLYGONS:
        meta = TIER_META[p["tier"]]
        out.append({
            "tier":       p["tier"],
            "name":       p["name"],
            "desc":       p["desc"],
            "lons":       p["lons"],
            "lats":       p["lats"],
            "line_color": meta["color"],
            "fill_color": meta["fill"],
            "label":      meta["label"],
        })
    return out


def get_fuel_zones_for_plotly() -> list[dict]:
    out = []
    for p in _FUEL_POLYGONS:
        out.append({
            "type": p["type"],
            "desc": p["desc"],
            "lons": p["lons"],
            "lats": p["lats"],
        })
    return out


def get_fuel_type_descriptions() -> dict[str, str]:
    """Return {fuel_type: description} for each unique fuel type present."""
    seen: dict[str, str] = {}
    for p in _FUEL_POLYGONS:
        if p["type"] not in seen:
            seen[p["type"]] = p["desc"]
    return seen


def annotate_buses(buses: dict) -> dict[str, dict]:
    """Return {bus_name: {tier, fuel_type}} for all buses with coordinates."""
    out: dict[str, dict] = {}
    for b, d in buses.items():
        lon, lat = d.get("x", 0.0), d.get("y", 0.0)
        out[b] = {
            "tier":      get_risk_tier(lon, lat),
            "fuel_type": get_fuel_type(lon, lat),
        }
    return out
