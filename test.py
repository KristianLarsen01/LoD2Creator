import geopandas as gpd
import fiona
from pprint import pprint

GML_PATH = "data/Basisdata_5001_Trondheim_5972_FKB-Bygning_GML.gml"

def print_layer_info(layer_name):
    print("\n" + "="*80)
    print(f"🔍 LAYER: {layer_name}")
    print("="*80)

    # --- 1) Les rått med Fiona (OGR) ---
    with fiona.open(GML_PATH, layer=layer_name) as src:
        print(f"Antall features: {len(src)}")
        print("Schema:")
        pprint(src.schema)

        # Finn første feature som faktisk har geometri
        first_feat = None
        for feat in src:
            if feat is not None and feat.get("geometry") is not None:
                first_feat = feat
                break

        print("\nEksempel på RÅ feature fra Fiona (før GeoPandas):")
        if first_feat is None:
            print("Ingen feature med geometri funnet i dette laget.")
        else:
            pprint(first_feat)
            geom = first_feat["geometry"]
            print("\nOGR geometry type:", geom["type"])
            # Bare print litt coords, så det ikke blir vegg av tall
            coords = geom.get("coordinates")
            print("OGR coordinates (truncert):")
            if isinstance(coords, (list, tuple)) and len(coords) > 5:
                pprint(coords[:5])
                print("... (flere koordinater)")
            else:
                pprint(coords)

    # --- 2) Les med GeoPandas ---
    print("\nLeser laget med GeoPandas...")
    gdf = gpd.read_file(GML_PATH, layer=layer_name)
    print("Kolonner:", list(gdf.columns))
    print("GeoPandas geom_type:", gdf.geom_type.unique())

    print("\nFørste rad i GeoPandas:")
    print(gdf.iloc[0])

    # --- 3) Telle geometri-typer ---
    print("\nGeometri-fordeling:")
    for t in ["Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"]:
        n = (gdf.geom_type == t).sum()
        print(f"  {t}: {n}")

    return gdf

def main():
    print("📌 LISTER ALLE LAG I GML-FILA\n")
    layers = fiona.listlayers(GML_PATH)
    pprint(layers)

    print("\n📌 SJEKKER LAG 'Bygning':")
    byg = print_layer_info("Bygning")

    print("\n📌 SJEKKER LAG 'AnnenBygning':")
    annen = print_layer_info("AnnenBygning")

if __name__ == "__main__":
    main()
