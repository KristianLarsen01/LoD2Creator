#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from shapely.geometry import Polygon, MultiPolygon, LineString, Point

from models.helpers import segment_ids_from_rgb
from models.gabled import build_gabled_roof


def ensure_polygon_3d(poly, default_z: float) -> Polygon:
    """
    Sørg for at vi har en Polygon med (x,y,z)-coords.
    - Hvis MultiPolygon: ta den største.
    - Hvis LineString/Point: buffer litt for å få en flate.
    - Hvis 2D: legg på default_z som høyde.
    """
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    elif isinstance(poly, LineString):
        poly = poly.buffer(0.05)
    elif isinstance(poly, Point):
        poly = poly.buffer(0.05)

    if not isinstance(poly, Polygon):
        poly = Polygon(poly)

    coords = np.array(poly.exterior.coords)

    # Hvis bare x,y: legg på default_z
    if coords.shape[1] == 2:
        x = coords[:, 0]
        y = coords[:, 1]
        z = np.full_like(x, default_z, dtype=float)
        coords3d = np.vstack([x, y, z]).T
        poly3d = Polygon(coords3d)
        return poly3d

    return poly


def plot_roof(points: np.ndarray, polys: dict[int, Polygon], title: str = ""):
    """
    Lager en figur med:
      - venstre: 2D (x,y) punktsky + polygon-outline
      - høyre: 3D punktsky + polygonflater
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Sentrer i XY for penere plott
    x_mean = x.mean()
    y_mean = y.mean()
    x0 = x - x_mean
    y0 = y - y_mean
    z_mean = z.mean()

    fig = plt.figure(figsize=(12, 6))

    # --- 2D topp-view ---
    ax1 = fig.add_subplot(1, 2, 1)
    sc = ax1.scatter(x0, y0, c=z, s=10, linewidths=0, cmap="viridis")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Punktsky (XY, farget etter Z)")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Lås aksegrenser til punktskyen, slik at polygonet ikke ødelegger skalaen
    ax1.set_xlim(x0.min(), x0.max())
    ax1.set_ylim(y0.min(), y0.max())

    # Overlay polygon-outline + hjørnepunkter i rødt
    for seg_id, poly in polys.items():
        poly3d = ensure_polygon_3d(poly, default_z=z_mean)
        coords = np.array(poly3d.exterior.coords)
        px = coords[:, 0] - x_mean
        py = coords[:, 1] - y_mean
        ax1.plot(px, py, "-r", linewidth=2)
        ax1.scatter(px, py, c="red", s=20)  # marker verticene tydelig

    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Z (høyde)")

    # --- 3D plott ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Punktsky i lys grå
    ax2.scatter(x0, y0, z, s=5, c="lightgrey")

    # Polygonflater i rødt
    for seg_id, poly in polys.items():
        poly3d = ensure_polygon_3d(poly, default_z=z_mean)
        coords = np.array(poly3d.exterior.coords)
        px = coords[:, 0] - x_mean
        py = coords[:, 1] - y_mean
        pz = coords[:, 2]
        verts = [list(zip(px, py, pz))]
        coll = Poly3DCollection(verts, alpha=0.7, facecolor="red", edgecolor="k")
        ax2.add_collection3d(coll)
        # markér også hjørnepunkter
        ax2.scatter(px, py, pz, c="red", s=30)

    ax2.set_title("Takpolygon(er) i 3D")
    ax2.set_xlabel("X (sentrert)")
    ax2.set_ylabel("Y (sentrert)")
    ax2.set_zlabel("Z")

    # Litt fornuftige aksegrenser
    max_range = np.array(
        [x0.max() - x0.min(), y0.max() - y0.min(), z.max() - z.min()]
    ).max() / 2.0
    mid_x = 0.0
    mid_y = 0.0
    mid_z = z_mean
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Bruk: python inspect_roof.py path/to/roof.laz")
        sys.exit(1)

    laz_path = Path(sys.argv[1]).expanduser().resolve()
    if not laz_path.is_file():
        print(f"Fant ikke fil: {laz_path}")
        sys.exit(1)

    print(f"Leser {laz_path} ...")
    las = laspy.read(laz_path)
    points = np.vstack((las.x, las.y, las.z)).T
    seg_ids = segment_ids_from_rgb(las)
    global_min_z = float(points[:, 2].min())

    try:
        seg_poly_map = build_gabled_roof(points, seg_ids, global_min_z)
    except Exception as e:
        print(f"FEIL ved bygging av gabled roof for {laz_path.name}: {e}")
        sys.exit(1)

    print(f"Fikk {len(seg_poly_map)} polygon(er) for {laz_path.name}")
    for sid, poly in seg_poly_map.items():
        print(f"  segment {sid}: geom_type={poly.geom_type}")

    plot_roof(points, seg_poly_map, title=laz_path.name)


if __name__ == "__main__":
    main()
