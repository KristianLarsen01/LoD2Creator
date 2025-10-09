#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vis ÉN LAZ-fil fra en mappe.

Eksempler:
  python view_one_laz.py sample_roofdata_50
  python view_one_laz.py sample_roofdata_50 --index 0
  python view_one_laz.py sample_roofdata_50 --file 182172235.laz
  python view_one_laz.py sample_roofdata_50 --file 1722      # substring-match
  python view_one_laz.py sample_roofdata_50 --autoscale --point-size 4
  
  python view_one_laz.py sample_roofdata_50 --index 3 --autoscale --point-size 4


Tips:
  - Hvis du får "No LazBackend selected, cannot decompress data": installer laspy med lazrs/laszip-backend.
"""

import argparse
import glob
import os
import sys
import numpy as np
import laspy
import open3d as o3d


def list_laz_files(folder, pattern="*.laz"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    return files


def select_file(files, index=None, name_or_substr=None):
    if not files:
        raise FileNotFoundError("Fant ingen .laz-filer i mappen.")

    if name_or_substr:
        # Prøv eksakt filnavn først
        base_map = {os.path.basename(f): f for f in files}
        if name_or_substr in base_map:
            return base_map[name_or_substr]

        # Ellers: substring-søk
        candidates = [f for f in files if name_or_substr in os.path.basename(f)]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            print("Fant flere treff for '--file':")
            for i, f in enumerate(candidates):
                print(f"  [{i}] {os.path.basename(f)}")
            print("Spesifiser mer av navnet, eller bruk --index.")
            sys.exit(1)
        else:
            print(f"Ingen filer matchet '{name_or_substr}'. Tilgjengelige filer:")
            for i, f in enumerate(files):
                print(f"  [{i}] {os.path.basename(f)}")
            sys.exit(1)

    # Fallback: bruk index (default 0)
    if index is None:
        index = 0
    if index < 0 or index >= len(files):
        raise IndexError(f"--index {index} er utenfor rekkevidde (0..{len(files)-1}).")
    return files[index]


def read_points(filepath):
    with laspy.open(filepath) as f:
        las = f.read()
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    return pts


def color_by_height(z):
    z = np.asarray(z, dtype=np.float64)
    zmin, zmax = float(np.min(z)), float(np.max(z))
    if zmax - zmin < 1e-9:
        t = np.zeros_like(z, dtype=np.float64)
    else:
        t = ((z - zmin) / (zmax - zmin)).astype(np.float64)
    cols = np.column_stack([t, 1.0 - t, 0.5 * (1.0 - t)]).astype(np.float64)
    return cols


def build_cloud(points, voxel=None):
    pts = np.ascontiguousarray(points, dtype=np.float64)
    cols = np.ascontiguousarray(color_by_height(pts[:, 2]), dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    return pcd


def center_and_scale(pts, do_center=True, autoscale=False):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    extent = mx - mn
    scale = 1.0
    if autoscale:
        max_extent = float(np.max(extent)) if np.max(extent) > 0 else 1.0
        scale = 1.0 / max_extent
    T = -center if do_center else np.zeros(3, dtype=np.float64)
    return T.astype(np.float64), float(scale), center.astype(np.float64), extent.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description="Visualiser én .laz-fil fra en mappe.")
    ap.add_argument("folder", help="Mappe med .laz-filer")
    ap.add_argument("--pattern", default="*.laz", help="Glob-mønster (default: *.laz)")
    ap.add_argument("--index", type=int, default=None, help="Indeks i sortert fil-liste (default: 0)")
    ap.add_argument("--file", dest="name_or_substr", help="Filnavn eller substring for å velge fil")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel-nedprøving (f.eks. 0.2). 0=av.")
    ap.add_argument("--autoscale", action="store_true", help="Skaler til enhetsterning for stabil visning.")
    ap.add_argument("--no-center", action="store_true", help="Ikke flytt til origo før visning (behold originale coords).")
    ap.add_argument("--point-size", type=float, default=3.0, help="Punktstørrelse i viewer.")
    args = ap.parse_args()

    files = list_laz_files(args.folder, pattern=args.pattern)
    chosen = select_file(files, index=args.index, name_or_substr=args.name_or_substr)

    print("=== Tilgjengelige filer (sortert) ===")
    for i, f in enumerate(files):
        mark = " <--" if f == chosen else ""
        print(f"[{i:02d}] {os.path.basename(f)}{mark}")

    print(f"\n[INFO] Leser: {chosen}")
    pts = read_points(chosen)
    if pts.size == 0:
        print("Ingen punkter i valgt fil.")
        sys.exit(1)

    # Transform for stabil visning
    T, S, center, extent = center_and_scale(pts, do_center=not args.no_center, autoscale=args.autoscale)
    print(f"[INFO] translate={T}, scale={S:.6f}, extent(before)={extent}")

    pts_tf = (pts + T) * S
    pcd = build_cloud(pts_tf, voxel=args.voxel if args.voxel and args.voxel > 0 else None)

    # Viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"LAZ Viewer - {os.path.basename(chosen)}", width=1280, height=800, visible=True)
    vis.add_geometry(pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 if args.autoscale else 1.0)
    vis.add_geometry(axis)

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1.0, 0.0, 0.0)
    vis.add_geometry(aabb)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0])

    ctr = vis.get_view_control()
    ctr.set_lookat(aabb.get_center())
    ctr.set_front([0, -1, -1])
    ctr.set_up([0, 0, 1])

    diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    if diag < 1e-6:   ctr.set_zoom(0.5)
    elif diag < 5:    ctr.set_zoom(0.6)
    elif diag < 50:   ctr.set_zoom(0.8)
    else:             ctr.set_zoom(0.9)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    try:
        main()
    except laspy.errors.LaspyException as e:
        print("\n[ERROR] laspy kunne ikke lese LAZ (mangler backend?).")
        print("Installer én av disse i miljøet ditt:")
        print('  pip install "laspy[lazrs]"   # anbefalt på macOS/Linux')
        print('  pip install "laspy[laszip]"  # alternativ backend')
        print(f"Detaljer: {e}")
        sys.exit(1)
