# models/helpers.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from shapely.ops import triangulate, unary_union
from shapely.geometry import MultiPoint, Polygon, LineString

import xml.etree.ElementTree as ET


# ---------- Plan- og snitt-hjelpere ----------

def fit_plane_least_squares(x, y, z) -> tuple[float, float, float, float]:
    """
    Fitter et plan Ax + By + Cz + D = 0 til punkter (x, y, z) med least squares.
    Normaliserer slik at ||(A,B,C)|| = 1.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    A_mat = np.c_[x, y, np.ones_like(x)]
    coeffs, *_ = np.linalg.lstsq(A_mat, z, rcond=None)
    a, b, c0 = coeffs  # z = a x + b y + c0

    # z - a x - b y - c0 = 0  =>  a x + b y - 1*z + c0 = 0
    A, B, C, D = a, b, -1.0, c0

    n = np.array([A, B, C], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        return 0.0, 0.0, 0.0, 0.0

    A, B, C, D = A / norm, B / norm, C / norm, D / norm
    return float(A), float(B), float(C), float(D)


def intersect_2planes(p1, p2):
    """
    Skjæringslinje (punkt + retningsvektor) mellom to plan.
    Plan er (A,B,C,D): A x + B y + C z + D = 0.
    Returnerer (point_on_line, direction_vector).
    """
    A1, B1, C1, D1 = p1
    A2, B2, C2, D2 = p2

    n1 = np.array([A1, B1, C1], dtype=float)
    n2 = np.array([A2, B2, C2], dtype=float)

    direction = np.cross(n1, n2)
    norm_d = np.linalg.norm(direction)
    if norm_d == 0:
        raise ValueError("Parallelle plan – ingen unik skjæringslinje.")
    direction = direction / norm_d

    # Finn ett punkt på linjen ved å velge z=0 og løse for x,y; fall tilbake til x=0 om nødvendig
    A = np.array([[A1, B1],
                  [A2, B2]], dtype=float)
    b = np.array([-D1, -D2], dtype=float)

    try:
        xy = np.linalg.solve(A, b)
        point = np.array([xy[0], xy[1], 0.0], dtype=float)
    except np.linalg.LinAlgError:
        A = np.array([[B1, C1],
                      [B2, C2]], dtype=float)
        b = np.array([-D1, -D2], dtype=float)
        yz = np.linalg.solve(A, b)
        point = np.array([0.0, yz[0], yz[1]], dtype=float)

    return point, direction


def intersect_3planes(p1, p2, p3):
    """
    Skjæringspunkt mellom tre plan (forutsatt at det finnes).
    Returnerer np.array([x,y,z]).
    """
    A = np.array([[p1[0], p1[1], p1[2]],
                  [p2[0], p2[1], p2[2]],
                  [p3[0], p3[1], p3[2]]], dtype=float)
    b = -np.array([p1[3], p2[3], p3[3]], dtype=float)

    xyz = np.linalg.solve(A, b)
    return xyz


def find_gabled_pair(
    planes: list[tuple[float, float, float, float]],
    max_ridge_slope_deg: float = 10.0,
    min_opposite_xy_angle_deg: float = 100.0,
) -> tuple[int, int] | None:
    """
    Finn ett hoved-gavlpar (i, j) blant planene.

    Kriterier:
      - Skjæringslinja (mønen) er nesten horisontal (liten z-komponent).
      - Normalene peker omtrent motsatt vei i xy (vinkel > min_opposite_xy_angle_deg).
    """
    best_pair = None
    best_score = -math.inf

    min_cos = math.cos(math.radians(min_opposite_xy_angle_deg))

    n = len(planes)
    for i in range(n):
        A1, B1, C1, _ = planes[i]
        n1 = np.array([A1, B1, C1], float)
        n1_xy = n1[:2]
        norm1_xy = np.linalg.norm(n1_xy)
        if norm1_xy == 0:
            continue

        for j in range(i + 1, n):
            A2, B2, C2, _ = planes[j]
            n2 = np.array([A2, B2, C2], float)
            n2_xy = n2[:2]
            norm2_xy = np.linalg.norm(n2_xy)
            if norm2_xy == 0:
                continue

            v = np.cross(n1, n2)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            v = v / v_norm

            ridge_slope = math.degrees(
                math.atan2(abs(v[2]), math.sqrt(v[0] ** 2 + v[1] ** 2))
            )
            if ridge_slope > max_ridge_slope_deg:
                continue

            cos_xy = float(np.dot(n1_xy, n2_xy) / (norm1_xy * norm2_xy))
            if cos_xy >= min_cos:
                continue

            # Score: nær horisontal møne + normaler godt motsatt
            score = -abs(cos_xy) - ridge_slope
            if score > best_score:
                best_score = score
                best_pair = (i, j)

    return best_pair


# ---------- Segmentering fra RGB ----------

def segment_ids_from_rgb(las) -> np.ndarray:
    """
    Lager en unik segment-ID per (R,G,B)-kombinasjon.
    Hvis filen ikke har RGB, returneres én felles ID for alle punkt.
    """
    dims = {d.lower() for d in las.point_format.dimension_names}
    n = len(las.x)

    if {"red", "green", "blue"}.issubset(dims):
        r = las.red.astype(np.int64)
        g = las.green.astype(np.int64)
        b = las.blue.astype(np.int64)
        seg_ids = (r << 16) + (g << 8) + b
    else:
        seg_ids = np.zeros(n, dtype=np.int64)

    return seg_ids


# ---------- CityGML-hjelpere ----------

def polygon_to_poslist_3d(poly):
    """
    Støtter Polygon, men hvis LineString eller Point kommer inn, 
    lag en minimal "flat polygon" som ikke crasher CityGML.
    """
    geom_type = poly.geom_type

    if geom_type == "Polygon":
        coords = list(poly.exterior.coords)
    elif geom_type == "LineString":
        # Lag en pseudo-2D polygon ved å bufre litt
        poly = poly.buffer(0.01)  # gjør LineString om til tynn polygon
        coords = list(poly.exterior.coords)
    elif geom_type == "Point":
        # Lag en liten sirkel rundt punktet
        poly = poly.buffer(0.1)
        coords = list(poly.exterior.coords)
    else:
        raise ValueError(f"polygon_to_poslist_3d: Unsupported geometry: {geom_type}")

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    flat = []
    for x, y, *rest in coords:
        z = rest[0] if rest else 0.0
        flat.extend([x, y, z])

    return " ".join(f"{v:.3f}" for v in flat)

def write_citygml(
    roofs: List[Dict[str, Any]],
    out_path: Path,
    srs_name: str = "urn:ogc:def:crs:EPSG::25832",
) -> None:
    """
    Skriver en veldig enkel CityGML 2.0 med Buildings og RoofSurfaces.

    roofs: liste med dict-er:
        {
          "roof_id": str,
          "polygons": [Polygon, Polygon, ...]
        }
    """
    ns = {
        "gml": "http://www.opengis.net/gml",
        "core": "http://www.opengis.net/citygml/2.0",
        "bldg": "http://www.opengis.net/citygml/building/2.0",
    }

    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    citymodel = ET.Element("{%s}CityModel" % ns["core"])
    citymodel.set("{%s}id" % ns["gml"], "cm_1")

    for roof in roofs:
        roof_id = roof["roof_id"]
        polys: list[Polygon] = roof["polygons"]

        com = ET.SubElement(citymodel, "{%s}cityObjectMember" % ns["core"])
        bldg_el = ET.SubElement(
            com,
            "{%s}Building" % ns["bldg"],
            {"{%s}id" % ns["gml"]: f"b_{roof_id}"},
        )

        for i, poly in enumerate(polys):
            bbounded = ET.SubElement(bldg_el, "{%s}boundedBy" % ns["bldg"])
            rs = ET.SubElement(
                bbounded,
                "{%s}RoofSurface" % ns["bldg"],
                {"{%s}id" % ns["gml"]: f"rs_{roof_id}_{i}"},
            )
            lod2 = ET.SubElement(rs, "{%s}lod2MultiSurface" % ns["bldg"])
            ms = ET.SubElement(
                lod2,
                "{%s}MultiSurface" % ns["gml"],
                {"srsName": srs_name},
            )
            sm = ET.SubElement(ms, "{%s}surfaceMember" % ns["gml"])
            poly_el = ET.SubElement(
                sm,
                "{%s}Polygon" % ns["gml"],
                {"{%s}id" % ns["gml"]: f"poly_{roof_id}_{i}"},
            )
            exterior = ET.SubElement(poly_el, "{%s}exterior" % ns["gml"])
            lr = ET.SubElement(exterior, "{%s}LinearRing" % ns["gml"])
            poslist = ET.SubElement(lr, "{%s}posList" % ns["gml"])
            poslist.text = polygon_to_poslist_3d(poly)

    tree = ET.ElementTree(citymodel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Skrev CityGML til: {out_path}")


def _simplify_hull_by_angle(hull: Polygon, straight_tol_deg: float = 5.0) -> Polygon:
    """
    Forenkler et konvekst hull ved å fjerne hjørner der den indre vinkelen
    er nesten 180° (dvs. punktet ligger omtrent på en rett linje).

    straight_tol_deg: hvor mange grader vi tillater at vinkelen avviker
                      fra 180° før vi beholder hjørnet.
    """
    coords = list(hull.exterior.coords)[:-1]  # uten duplikat siste
    n = len(coords)
    if n <= 4:
        # allerede trekant/rektangel – ingenting å gjøre
        return hull

    new_pts = []
    for i in range(n):
        x_prev, y_prev = coords[(i - 1) % n]
        x_cur,  y_cur  = coords[i]
        x_next, y_next = coords[(i + 1) % n]

        v_in  = np.array([x_prev - x_cur, y_prev - y_cur], dtype=float)
        v_out = np.array([x_next - x_cur, y_next - y_cur], dtype=float)

        if np.linalg.norm(v_in) == 0 or np.linalg.norm(v_out) == 0:
            continue

        # indre vinkel i hjørnet
        dot = np.dot(v_in, v_out) / (np.linalg.norm(v_in) * np.linalg.norm(v_out))
        dot = float(np.clip(dot, -1.0, 1.0))
        angle = np.degrees(np.arccos(dot))  # 0 = skarp, 180 = rett linje

        # hvis (nesten) 180°, dropp dette punktet
        if abs(angle - 180.0) <= straight_tol_deg:
            continue

        new_pts.append((x_cur, y_cur))

    if len(new_pts) < 3:
        # ble for aggressiv – gå tilbake til original
        return hull

    new_pts.append(new_pts[0])  # lukk polygon
    return Polygon(new_pts)


def compute_roof_outline_2d(
    points: np.ndarray,
    buffer_eps: float = 0.05,
    mode: str = "convex",
    edge_factor: float = 0.15,
) -> Polygon:
    """
    Beregn et 2D-omriss (Polygon i XY) for ett tak.

    mode:
      - "convex": konveks hull (MultiPoint(...).convex_hull)
      - "concave": konkav hull (triangulerings-basert)
      - "rectangle": minimum rotert rektangel

    edge_factor brukes i "concave"-modus: hvor detaljert omrisset
    kan bli relativt til byggets størrelse.
    """
    points = np.asarray(points)
    if points.shape[0] == 0:
        raise ValueError("compute_roof_outline_2d: Ingen punkter gitt.")

    xy = points[:, :2]
    mp = MultiPoint(xy)

    if mode == "rectangle":
        hull = mp.convex_hull
        if not isinstance(hull, Polygon):
            hull = hull.buffer(buffer_eps)
        return hull.minimum_rotated_rectangle

    if mode == "concave":
        return _concave_outline_from_points(xy, edge_factor=edge_factor, buffer_eps=buffer_eps)

    # default: convex
    hull = mp.convex_hull
    if not isinstance(hull, Polygon):
        hull = hull.buffer(buffer_eps)
    return hull



def lift_outline_to_z(outline_2d: Polygon, z0: float) -> Polygon:
    """
    Løfter et 2D-omriss (Polygon i XY) til en 3D-Polygon
    ved å sette alle hjørner til høyde z0.

    Dette er midlertidig – senere vil du erstatte z0 med
    skjæringshøyder mellom plan og “fotplanet”.
    """
    if outline_2d.is_empty:
        raise ValueError("lift_outline_to_z: Tomt polygon.")

    coords2d = list(outline_2d.exterior.coords)

    coords3d = [(x, y, z0) for (x, y) in coords2d]
    poly3d = Polygon(coords3d)
    return poly3d

def _concave_outline_from_points(
    points_xy: np.ndarray,
    edge_factor: float = 0.15,
    buffer_eps: float = 0.05,
) -> Polygon:
    """
    Konkavt omriss (alpha-shape-lignende) basert på Delaunay-triangulering.

    edge_factor: styrer hvor "tannete" omrisset kan bli.
      - liten verdi  -> glattere (nær konveks)
      - stor verdi   -> mer detaljert, flere knekk

    points_xy: [N,2] array med (x,y)
    """
    mp = MultiPoint(points_xy)
    if len(mp.geoms) == 0:
        raise ValueError("_concave_outline_from_points: ingen punkter")

    # Grov skala på bygget (diagonal av bounding box)
    mins = points_xy.min(axis=0)
    maxs = points_xy.max(axis=0)
    bb_diag = float(np.linalg.norm(maxs - mins))
    if bb_diag == 0:
        # alle punkt likt -> lite buffer
        return mp.buffer(buffer_eps)

    max_edge = bb_diag * edge_factor

    # Delaunay-triangulering av punktmengden
    tris = triangulate(mp)
    if not tris:
        # fallback: konveks hull
        return mp.convex_hull

    kept = []
    for tri in tris:
        coords = list(tri.exterior.coords)[:3]  # tre hjørner
        edges = []
        for i in range(3):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % 3]
            edges.append(np.hypot(x2 - x1, y2 - y1))
        if max(edges) <= max_edge:
            kept.append(tri)

    # Hvis vi kastet alt (for streng terskel) -> beholde den største trekanten
    if not kept:
        kept = [max(tris, key=lambda t: t.area)]

    merged = unary_union(kept)

    if isinstance(merged, Polygon):
        poly = merged
    else:
        # MultiPolygon: ta den største
        poly = max(list(merged.geoms), key=lambda g: g.area)

    # buffer(0) for å rydde opp eventuelle små topologifeil
    poly = poly.buffer(0)
    if poly.is_empty:
        return mp.convex_hull

    return poly
