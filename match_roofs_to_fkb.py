import glob
import os

import laspy
import geopandas as gpd
from shapely.geometry import box

# ---- KONFIG ----

LAZ_DIR = "data/sample_roofdata_50"
FKB_GML = "data/Basisdata_5001_Trondheim_5972_FKB-Bygning_GML.gml"
FKB_LAYER = "AnnenBygning"          # polygon Z
OUTPUT_GML = "data/FKB_LoD2_50hus.gml"

LAZ_CRS = "EPSG:25832"
MAX_DIST = 50.0      # meter – kan økes/endres etter du ser resultatene


def build_roof_bboxes(laz_dir: str, crs):
    paths = sorted(glob.glob(os.path.join(laz_dir, "*.laz")))
    if not paths:
        raise FileNotFoundError(f"Ingen .laz i {laz_dir!r}")

    filenames = []
    geoms = []
    for p in paths:
        las = laspy.read(p)
        minx, miny, _ = las.header.min
        maxx, maxy, _ = las.header.max
        filenames.append(os.path.basename(p))
        geoms.append(box(minx, miny, maxx, maxy))

    return gpd.GeoDataFrame({"filename": filenames, "geometry": geoms}, crs=crs)


def main():
    print(f"Leser footprint-lag '{FKB_LAYER}' ...")
    fkb = gpd.read_file(FKB_GML, layer=FKB_LAYER)
    print("Geom-typer:", fkb.geom_type.unique())
    print("Antall footprints totalt:", len(fkb))

    roofs = build_roof_bboxes(LAZ_DIR, LAZ_CRS)
    print("Antall tak:", len(roofs))
    print("Tak bounds:", roofs.total_bounds)

    # Klipp ned FKB til området rundt takene + buffer
    minx, miny, maxx, maxy = roofs.total_bounds
    buffer = 50
    study_poly = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    study_gdf = gpd.GeoDataFrame(geometry=[study_poly], crs=LAZ_CRS)

    fkb_clip = gpd.overlay(fkb, study_gdf, how="intersection")
    print("Footprints i studieområdet:", len(fkb_clip))

    if fkb_clip.empty:
        print("Ingen footprints i området – sjekk koordinater.")
        return

    # Lag punktrepresentasjon av takene (centroid av bbox)
    roof_pts = roofs.copy()
    roof_pts["geometry"] = roof_pts.geometry.centroid

    print("Kjører sjoin_nearest (nærmeste footprint per tak) ...")
    joined = gpd.sjoin_nearest(
        roof_pts,
        fkb_clip,
        how="left",
        distance_col="dist"
    )

    print("\nEksempel (før filtrering på max_dist):")
    print(joined[["filename", "dist"]].head())

    # Filtrer bort åpenbart feil matcher (for langt unna)
    joined_near = joined[joined["dist"] <= MAX_DIST].copy()
    print(f"\nTak med footprint innen {MAX_DIST} m:", len(joined_near))

    matched_idx = joined_near["index_right"].dropna().unique().astype(int)
    print("Unike matchende footprints:", len(matched_idx))

    if len(matched_idx) == 0:
        print("Fant ingen bygg innen maksavstand – prøv større MAX_DIST.")
        return

    footprints = fkb_clip.loc[matched_idx]
    print(f"Skriver ut {len(footprints)} bygg til '{OUTPUT_GML}' ...")
    footprints.to_file(OUTPUT_GML, driver="GML")
    print("Ferdig! 🚀")


if __name__ == "__main__":
    main()
