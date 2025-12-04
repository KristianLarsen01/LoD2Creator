import os
import glob

import laspy
import numpy as np
import matplotlib.pyplot as plt

# Folder with your 50 roofs
DATA_DIR = "data/Weird_gabled"

# OPTIONAL: import your guess function here, if you have one
# from your_module import guess_roof_type


def visualize_roof(laz_path, guessed_type=None):
    las = laspy.read(laz_path)

    # Convert ScaledArrayView -> numpy arrays
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    # Skip empty files
    if x.size == 0:
        print(f"⚠️  File {laz_path} has zero points — skipping.")
        return

    # Center in XY for nicer visualization
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    z0 = z - np.min(z)  # height shift so min z = 0

    fig, ax = plt.subplots(figsize=(5, 5))

    sc = ax.scatter(x0, y0, c=z0, s=1, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (relative)")

    fname = os.path.basename(laz_path)
    if guessed_type is None:
        title = fname
    else:
        title = f"{fname} — guessed: {guessed_type}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def main():
    laz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.laz")))

    print(f"Found {len(laz_files)} roofs")
    if len(laz_files) == 0:
        print("No .laz files found — check DATA_DIR.")
        return

    for laz_path in laz_files:
        # Plug in your guess function here if you have one:
        # guessed_type = guess_roof_type(laz_path)
        guessed_type = None
        visualize_roof(laz_path, guessed_type=guessed_type)


if __name__ == "__main__":
    main()
