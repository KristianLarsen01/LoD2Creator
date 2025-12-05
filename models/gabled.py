# models/gabled.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import polygonize

from .helpers import (
    fit_plane_least_squares,
    intersect_2planes,
    intersect_3planes,
    find_gabled_pair,
)


def build_gabled_roof(
    points: np.ndarray,
    segment_ids: np.ndarray,
    global_min_z: Optional[float] = None,
) -> Dict[int, Polygon]:
    """
    Bygg polygoner for et (enkelt) gavltak.

    Parametre
    ---------
    points : [N,3] XYZ for alle punkt (ett tak).
    segment_ids : [N] segment-ID per punkt.
    global_min_z : om satt, brukes som "bakkenivå" for fotpunkter.

    Returnerer
    ---------
    dict: segment_id -> shapely Polygon
          Kun de to hoved-gavlsegmentene håndteres i denne versjonen.
    """
    points = np.asarray(points)
    segment_ids = np.asarray(segment_ids)

    unique_segments = np.unique(segment_ids)

    # 1) Plan per segment
    planes: list[tuple[float, float, float, float] | None] = []
    valid_mask: list[bool] = []
    for seg_id in unique_segments:
        mask = segment_ids == seg_id
        x, y, z = points[mask, 0], points[mask, 1], points[mask, 2]
        if len(x) < 3:
            planes.append(None)
            valid_mask.append(False)
            continue
        plane = fit_plane_least_squares(x, y, z)
        planes.append(plane)
        valid_mask.append(True)

    planes_valid = [p for p, ok in zip(planes, valid_mask) if ok]
    segs_valid = [sid for sid, ok in zip(unique_segments, valid_mask) if ok]

    if len(planes_valid) < 2:
        raise ValueError("For få gyldige plan til å lage gavltak.")

    pair_idx = find_gabled_pair(planes_valid)
    if pair_idx is None:
        raise ValueError("Fant ikke noe gavlpar i dette taket.")

    i_plane, j_plane = pair_idx
    p1 = planes_valid[i_plane]
    p2 = planes_valid[j_plane]
    seg_id_1 = segs_valid[i_plane]
    seg_id_2 = segs_valid[j_plane]

    # 2) Skjæringslinje (møne)
    ridge_point, ridge_dir = intersect_2planes(p1, p2)

    # 3) Utbredelse langs møneretningen i xy
    mask_1 = segment_ids == seg_id_1
    mask_2 = segment_ids == seg_id_2
    pts_12 = np.vstack([points[mask_1], points[mask_2]])

    ridge_dir_xy = np.array([ridge_dir[0], ridge_dir[1], 0.0], float)
    ridge_dir_xy /= np.linalg.norm(ridge_dir_xy)

    vecs = pts_12 - ridge_point
    t = vecs @ ridge_dir_xy
    t_min, t_max = t.min(), t.max()

    end1 = ridge_point + t_min * ridge_dir_xy
    end2 = ridge_point + t_max * ridge_dir_xy

    # 4) Vertikale kantplan gjennom endene (normal vinkelrett på mønen i xy)
    up = np.array([0.0, 0.0, 1.0])
    edge_normal = np.cross(ridge_dir_xy, up)
    edge_normal /= np.linalg.norm(edge_normal)

    def vertical_edge_plane(point: np.ndarray) -> np.ndarray:
        A, B, C = edge_normal
        x0, y0, z0 = point
        D = -(A * x0 + B * y0 + C * z0)
        return np.array([A, B, C, D], float)

    edge_plane1 = vertical_edge_plane(end1)
    edge_plane2 = vertical_edge_plane(end2)

    # 5) Topp-punkter langs mønen (skjæring av p1,p2 med hver kantplane)
    tip1 = intersect_3planes(p1, p2, edge_plane1)
    tip2 = intersect_3planes(p1, p2, edge_plane2)

    # 6) Fotplan (horisontalt plan på global_min_z eller min z i segmentene)
    def make_z_plane(z0: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0, -z0], float)

    if global_min_z is None:
        min_z = float(min(points[mask_1, 2].min(), points[mask_2, 2].min()))
    else:
        min_z = float(global_min_z)

    z_plane = make_z_plane(min_z)

    foot1_seg1 = intersect_3planes(p1, edge_plane1, z_plane)
    foot2_seg1 = intersect_3planes(p1, edge_plane2, z_plane)
    foot1_seg2 = intersect_3planes(p2, edge_plane1, z_plane)
    foot2_seg2 = intersect_3planes(p2, edge_plane2, z_plane)

    # 7) Polygoner (konveks hull av topp + fot)
    poly1 = MultiPoint(
        [tuple(tip1), tuple(tip2), tuple(foot1_seg1), tuple(foot2_seg1)]
    ).convex_hull
    poly2 = MultiPoint(
        [tuple(tip1), tuple(tip2), tuple(foot1_seg2), tuple(foot2_seg2)]
    ).convex_hull

    if poly1.geom_type != "Polygon":
        poly1 = poly1.buffer(0.05)  # gjør den til liten polygon hvis lineær
    if poly2.geom_type != "Polygon":
        poly2 = poly2.buffer(0.05)

    return {int(seg_id_1): poly1, int(seg_id_2): poly2}
