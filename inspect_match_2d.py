import geopandas as gpd
import laspy, glob
from shapely.geometry import box
import matplotlib.pyplot as plt

BUILDINGS = "data/FKB_LoD2_50hus.gml"
LAZ_DIR = "data/sample_roofdata_50"

def load_roofs(crs):
    geoms = []
    for p in glob.glob(f"{LAZ_DIR}/*.laz"):
        las = laspy.read(p)
        minx, miny, _ = las.header.min
        maxx, maxy, _ = las.header.max
        geoms.append(box(minx, miny, maxx, maxy))
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)

b = gpd.read_file(BUILDINGS)
r = load_roofs(b.crs)

ax = b.plot(edgecolor="black", facecolor="none")
r.plot(ax=ax, edgecolor="red", facecolor="none")

plt.gca().set_aspect("equal", "box")
plt.title("AnnenBygning (svart) + LAZ-tak-bbokser (rød)")
plt.show()
