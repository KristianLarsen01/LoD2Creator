#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import matplotlib.pyplot as plt

from models.helpers import compute_roof_outline_2d


DATA_ROOT = Path("data") / "gabled"


def inspect_roof_outline(laz_path: Path):
    print(f"\n=== Viser omriss for: {laz_path.name} ===")

    las = laspy.read(laz_path)
    points = np.vstack((las.x, las.y, las.z)).T

    if points.shape[0] == 0:
        print("  ⚠️  Filen har ingen punkter – hopper over.")
        return

    # XY og Z
    xy = points[:, :2]
    z = points[:, 2]

    # Sentrer i XY for penere plotting
    xy_mean = xy.mean(axis=0)
    xy_centered = xy - xy_mean

    # Beregn 2D-omriss i originalkoordinater
    outline_2d = compute_roof_outline_2d(
    points,
    mode="concave",
    edge_factor=0.15,  # juster ved behov
)
    outline_xy = np.asarray(outline_2d.exterior.coords)[:, :2]

    # Sentrer omrisset på samme måte
    outline_xy_centered = outline_xy - xy_mean

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(
        xy_centered[:, 0],
        xy_centered[:, 1],
        c=z,
        s=1,
        linewidths=0,
    )

    ax.plot(
        outline_xy_centered[:, 0],
        outline_xy_centered[:, 1],
        linewidth=2,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Høyde (z)")

    ax.set_title(f"{laz_path.name} – takomriss")

    plt.tight_layout()
    plt.show()


def main():
    if not DATA_ROOT.is_dir():
        print(f"Fant ikke mappe: {DATA_ROOT}")
        return

    laz_files = sorted(DATA_ROOT.glob("*.laz")) + sorted(DATA_ROOT.glob("*.las"))
    if not laz_files:
        print(f"Ingen .laz/.las-filer i {DATA_ROOT}")
        return

    print(f"Fant {len(laz_files)} filer i {DATA_ROOT}")
    print("Lukk figurvinduet for å gå videre til neste tak. Ctrl+C for å avbryte.\n")

    for laz_path in laz_files:
        try:
            inspect_roof_outline(laz_path)
        except KeyboardInterrupt:
            print("\nAvbrutt av bruker.")
            break
        except Exception as e:
            print(f"  FEIL under inspeksjon av {laz_path.name}: {e}")


if __name__ == "__main__":
    main()
