#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import laspy
import numpy as np
from shapely.geometry import Polygon

from models import gabled
from models.helpers import (
    segment_ids_from_rgb,
    write_citygml,
    compute_roof_outline_2d,
    lift_outline_to_z,
)


DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("outputs")


def process_category(
    category: str,
    builder: Callable[[np.ndarray, np.ndarray, float], Dict[int, Polygon]],
):
    """
    Leser alle .laz/.las i data/<category>/,
    bygger takpolygoner med 'builder',
    og skriver en CityGML for denne kategorien.
    """
    folder = DATA_ROOT / category
    if not folder.is_dir():
        print(f"Hopper over '{category}' – ingen mappe {folder}")
        return

    roofs_for_gml: List[Dict] = []

    laz_files = sorted(list(folder.glob("*.laz"))) + sorted(
        list(folder.glob("*.las"))
    )
    if not laz_files:
        print(f"Ingen .laz/.las i {folder}")
        return

    for laz_path in laz_files:
        print(f"[{category}] Behandler {laz_path.name} ...")
        las = laspy.read(laz_path)
        points = np.vstack((las.x, las.y, las.z)).T
        seg_ids = segment_ids_from_rgb(las)
        global_min_z = float(points[:, 2].min())

        # --- NYTT FELLES STEG: beregn omriss ---
        outline_2d = compute_roof_outline_2d(points)
        outline_3d = lift_outline_to_z(outline_2d, global_min_z)

        try:
            seg_poly_map = builder(points, seg_ids, global_min_z)
        except Exception as e:
            print(f"  FEIL i {laz_path.name}: {e}")
            continue

        polys = list(seg_poly_map.values())
        polys = [p if p.geom_type == "Polygon" else p.buffer(0.05) for p in polys]

        roofs_for_gml.append({"roof_id": laz_path.stem, "polygons": polys})

    if roofs_for_gml:
        out_path = OUTPUT_ROOT / f"{category}.gml"
        write_citygml(roofs_for_gml, out_path)

def build_outline_only(
    points: np.ndarray,
    segment_ids: np.ndarray,
    global_min_z: float,
) -> Dict[int, Polygon]:
    outline_2d = compute_roof_outline_2d(points)
    outline_3d = lift_outline_to_z(outline_2d, global_min_z)
    # Bruk f.eks. segment-id -1 for “omriss”
    return {-1: outline_3d}



def main():
    # Kommenter ut de kategoriene du ikke vil kjøre
    categories = [
        ("gabled", gabled.build_gabled_roof),
        ("outline_test", build_outline_only),
        # ("hipped", hipped.build_hipped_roof),
        # ("flat", flat.build_flat_roof),
        # ("complex_gabled", complex_gabled.build_complex_gabled_roof),
        # ("complex_hipped", complex_hipped.build_complex_hipped_roof),
    ]

    for cat_name, builder in categories:
        process_category(cat_name, builder)


if __name__ == "__main__":
    main()
