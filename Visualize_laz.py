#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#run source .venv/bin/activate 
#run command: python visualize_laz.py sample_roofdata_50 --autoscale --point-size 4 --color-by file --legend


import argparse
import glob
import os
import sys
import numpy as np
import laspy
import open3d as o3d


def read_points(filepath, class_filter=None):
    with laspy.open(filepath) as f:
        las = f.read()

    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    mask = None
    if class_filter is not None and hasattr(las, "classification"):
        mask = np.isin(las.classification, np.array(class_filter, dtype=np.uint8))
        pts = pts[mask]

    extras = {}
    if hasattr(las, "intensity"):
        inten = las.intensity.astype(np.float64)
        if mask is not None:
            inten = inten[mask]
        extras["intensity"] = inten

    return pts, extras


def color_by_height(z):
    z = np.asarray(z, dtype=np.float64)
    zmin, zmax = float(np.min(z)), float(np.max(z))
    if zmax - zmin < 1e-9:
        t = np.zeros_like(z, dtype=np.float64)
    else:
        t = ((z - zmin) / (zmax - zmin)).astype(np.float64)
    # R = t, G = 1 - t, B = 0.5*(1 - t)
    cols = np.column_stack([t, 1.0 - t, 0.5 * (1.0 - t)]).astype(np.float64)
    return cols


def color_by_intensity(intensity):
    inten = np.asarray(intensity, dtype=np.float64)
    imin, imax = float(np.min(inten)), float(np.max(inten))
    if imax - imin < 1e-9:
        t = np.zeros_like(inten, dtype=np.float64)
    else:
        t = ((inten - imin) / (imax - imin)).astype(np.float64)
    cols = np.column_stack([t, t, t]).astype(np.float64)  # gråskala
    return cols


def distinct_palette(n):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    hsv = np.zeros((n, 3), dtype=np.float64)
    hsv[:, 0] = (np.arange(n, dtype=np.float64) / max(n, 1))
    hsv[:, 1] = 0.85
    hsv[:, 2] = 0.95
    rgb = np.zeros_like(hsv)
    for i, (h, s, v) in enumerate(hsv):
        h6 = h * 6.0
        c = v * s
        x = c * (1 - abs((h6 % 2) - 1))
        m = v - c
        if   0 <= h6 < 1: rp, gp, bp = c, x, 0
        elif 1 <= h6 < 2: rp, gp, bp = x, c, 0
        elif 2 <= h6 < 3: rp, gp, bp = 0, c, x
        elif 3 <= h6 < 4: rp, gp, bp = 0, x, c
        elif 4 <= h6 < 5: rp, gp, bp = x, 0, c
        else:             rp, gp, bp = c, 0, x
        rgb[i] = [rp + m, gp + m, bp + m]
    return rgb.astype(np.float64)


def per_file_color(n_points, rgb):
    cols = np.tile(np.asarray(rgb, dtype=np.float64).reshape(1, 3), (n_points, 1))
    return cols


def build_cloud(pts, colors, voxel=None):
    # Sikkerhetsnett: dtype/shape/contiguity
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points må være (N,3), fikk {pts.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors må være (N,3), fikk {colors.shape}")
    if pts.shape[0] != colors.shape[0]:
        raise ValueError(f"antall punkter ({pts.shape[0]}) != antall farger ({colors.shape[0]})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    return pcd


def print_stats(name, pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    cnt = len(pts)
    diag = np.linalg.norm(mx - mn)
    print(f"[{name}] n={cnt:,}  xmin,ymin,zmin={mn}  xmax,ymax,zmax={mx}  diag={diag:.3f}")


def center_and_scale(all_pts, do_center=True, autoscale=False):
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    center = 0.5 * (mn + mx)
    extent = mx - mn
    scale = 1.0
    if autoscale:
        max_extent = float(np.max(extent)) if np.max(extent) > 0 else 1.0
        scale = 1.0 / max_extent
    T = -center if do_center else np.zeros(3, dtype=np.float64)
    return T.astype(np.float64), float(scale), center.astype(np.float64), extent.astype(np.float64)


def visualize(folder, voxel=None, class_filter=None, do_center=True, autoscale=False,
              point_size=2.0, color_by="file", show_legend=False):
    if not os.path.isdir(folder):
        print(f"ERROR: '{folder}' er ikke en mappe.")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(folder, "*.laz")))
    if not files:
        print(f"Fant ingen .laz-filer i '{folder}'.")
        sys.exit(1)

    per_file_pts, per_file_extras, kept_files = [], [], []
    for fpath in files:
        pts, extras = read_points(fpath, class_filter=class_filter)
        if pts.size == 0:
            print(f"[WARN] ingen punkter etter filtrering i: {fpath}")
            continue
        print_stats(os.path.basename(fpath), pts)
        per_file_pts.append(pts)
        per_file_extras.append(extras)
        kept_files.append(fpath)

    if not per_file_pts:
        print("Ingen punktskyer å vise.")
        sys.exit(1)

    # Felles transform
    all_pts = np.vstack(per_file_pts)
    T, S, center, extent = center_and_scale(all_pts, do_center=do_center, autoscale=autoscale)
    print(f"[INFO] translate={T}, scale={S:.6f}, extent(before)={extent}")

    clouds = []
    legend_rows = []

    if color_by == "file":
        palette = distinct_palette(len(per_file_pts))
        for i, (pts, extras) in enumerate(zip(per_file_pts, per_file_extras)):
            pts_tf = (pts + T) * S
            cols = per_file_color(len(pts_tf), palette[i])
            clouds.append(build_cloud(pts_tf, cols, voxel=voxel))
            legend_rows.append((os.path.basename(kept_files[i]), palette[i]))

    elif color_by == "height":
        for pts, extras in zip(per_file_pts, per_file_extras):
            pts_tf = (pts + T) * S
            cols = color_by_height(pts_tf[:, 2])
            clouds.append(build_cloud(pts_tf, cols, voxel=voxel))

    elif color_by == "intensity":
        for (pts, extras) in zip(per_file_pts, per_file_extras):
            pts_tf = (pts + T) * S
            if "intensity" in extras and len(extras["intensity"]) == len(pts):
                cols = color_by_intensity(extras["intensity"])
            else:
                print("[INFO] intensitet mangler/ulik lengde; faller tilbake til høyde-farge.")
                cols = color_by_height(pts_tf[:, 2])
            clouds.append(build_cloud(pts_tf, cols, voxel=voxel))

    else:
        print(f"[WARN] ukjent --color-by '{color_by}', bruker 'file'.")
        palette = distinct_palette(len(per_file_pts))
        for i, pts in enumerate(per_file_pts):
            pts_tf = (pts + T) * S
            cols = per_file_color(len(pts_tf), palette[i])
            clouds.append(build_cloud(pts_tf, cols, voxel=voxel))

    # Sammenslå for kamera-ramme
    merged = o3d.geometry.PointCloud()
    for c in clouds:
        merged += c
    aabb = merged.get_axis_aligned_bounding_box()
    aabb.color = (1.0, 0.0, 0.0)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LAZ Viewer", width=1280, height=800, visible=True)
    for c in clouds:
        vis.add_geometry(c)
    vis.add_geometry(aabb)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 if autoscale else 1.0)
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0])

    ctr = vis.get_view_control()
    lookat = aabb.get_center()
    ctr.set_lookat(lookat)
    ctr.set_front([0, -1, -1])
    ctr.set_up([0, 0, 1])

    diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    if diag < 1e-6:   ctr.set_zoom(0.5)
    elif diag < 5:    ctr.set_zoom(0.6)
    elif diag < 50:   ctr.set_zoom(0.8)
    else:             ctr.set_zoom(0.9)

    if show_legend and legend_rows:
        print("\n=== Filfarger (RGB 0-255) ===")
        for name, rgb in legend_rows:
            rgb255 = (np.clip(rgb, 0, 1) * 255).round().astype(int)
            print(f"{name:40s} -> {tuple(int(v) for v in rgb255)}")

    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description="Visualiser .laz-filer i en mappe.")
    ap.add_argument("folder", help="Mappe som inneholder .laz-filer")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel-nedprøving (f.eks. 0.2). 0=av.")
    ap.add_argument("--class", dest="class_filter", nargs="+", type=int, help="Klassifikasjonskode(r) å beholde.")
    ap.add_argument("--no-center", action="store_true", help="Ikke flytt til origo før visning.")
    ap.add_argument("--autoscale", action="store_true", help="Skaler til enhetsterning.")
    ap.add_argument("--point-size", type=float, default=2.0, help="Punktstørrelse (piksler).")
    ap.add_argument("--color-by", choices=["file", "height", "intensity"], default="file",
                    help="Fargestrategi: unik farge per fil, høyde, eller intensitet.")
    ap.add_argument("--legend", action="store_true", help="Skriv ut mapping fil -> farge.")
    args = ap.parse_args()

    visualize(
        args.folder,
        voxel=args.voxel if args.voxel and args.voxel > 0 else None,
        class_filter=args.class_filter,
        do_center=not args.no_center,
        autoscale=args.autoscale,
        point_size=args.point_size,
        color_by=args.color_by,
        show_legend=args.legend,
    )


if __name__ == "__main__":
    main()
