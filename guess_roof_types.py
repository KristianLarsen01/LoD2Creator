#!/usr/bin/env python
"""
Grovt skript for å gjette taktype for alle .laz-filer i en mappe.

- Antar én bygning (ett tak) per .laz-fil.
- Antar at takflater/segmenter er kodet via RGB-farge:
    hver unik (R,G,B) = ett segment.
- Fitter plan til hvert segment (least squares).
- Bruker enkle heuristikker for å gjette taktype:
    - Flat
    - Mono-pitch (ett skråplan)
    - Gabled
    - Complex_gabled
    - Hipped_or_pyramid
    - Unknown

Resultatet lagres i "roof_type_guesses.csv" i samme folder som skriptet.

Bruk:
    python guess_roof_types.py /sti/til/sample_roofdata_50
"""

import sys
from pathlib import Path
import math
import csv
from collections import defaultdict

import numpy as np
import laspy


# ---------------------------
# Geometri-hjelpefunksjoner
# ---------------------------

def fit_plane_least_squares(x, y, z):
    """
    Fitter et plan Ax + By + Cz + D = 0 til punkter (x, y, z) med least squares.
    Returnerer (A, B, C, D) normalisert slik at ||(A,B,C)|| = 1.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Design-matrise for z som avhengig variabel: z = ax + by + c
    # Vi fitter bare et "graf"-plan z(x,y), og konverterer det til Ax+By+Cz+D=0
    A_mat = np.c_[x, y, np.ones_like(x)]
    # minst-kvadraters-løsning
    coeffs, *_ = np.linalg.lstsq(A_mat, z, rcond=None)
    a, b, c0 = coeffs  # z = a x + b y + c0

    # Skriv om til A x + B y + C z + D = 0
    # z - a x - b y - c0 = 0  =>  a x + b y - 1*z + c0 = 0
    A = a
    B = b
    C = -1.0
    D = c0

    n = np.array([A, B, C], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        return 0.0, 0.0, 0.0, 0.0

    A /= norm
    B /= norm
    C /= norm
    D /= norm

    return float(A), float(B), float(C), float(D)


def slope_from_plane(plane):
    """
    Beregner helningsvinkel (i grader) for et plan.
    0° = helt flatt, større vinkel = brattere.
    """
    A, B, C, _ = plane
    horizontal_len = math.sqrt(A*A + B*B)
    if horizontal_len == 0:
        return 0.0
    # vinkel mellom planet og horisontal: tan(theta) = |hor| / |C|
    theta = math.atan2(horizontal_len, abs(C))
    return math.degrees(theta)


def find_gabled_pairs(planes,
                      max_ridge_slope_deg=5.0,
                      min_opposite_xy_angle_deg=120.0):
    """
    Leter etter "gavl-par" blant planene.
    Idé:
      - Intersection line mellom to plan har retning v = n1 x n2.
      - Hvis |v_z| er liten => linja nesten horisontal => kandidat for møne.
      - Hvis de horisontale normalene peker noenlunde mot hverandre
        (vinkel mellom n1_xy og n2_xy > min_opposite_xy_angle_deg),
        så er det typisk to takflater på hver side av en gavl.

    Returnerer liste av (i, j)-par med indeks i 'planes'.
    """
    pairs = []
    n = len(planes)
    if n < 2:
        return pairs

    min_cos = math.cos(math.radians(min_opposite_xy_angle_deg))
    # NB: vinkel > 120° betyr cos(vinkel) < cos(120°) = -0.5

    for i in range(n):
        A1, B1, C1, _ = planes[i]
        n1 = np.array([A1, B1, C1], dtype=float)
        n1_xy = n1[:2]
        norm1_xy = np.linalg.norm(n1_xy)
        if norm1_xy == 0:
            continue

        for j in range(i+1, n):
            A2, B2, C2, _ = planes[j]
            n2 = np.array([A2, B2, C2], dtype=float)
            n2_xy = n2[:2]
            norm2_xy = np.linalg.norm(n2_xy)
            if norm2_xy == 0:
                continue

            # Retning på interseksjons-linja
            v = np.cross(n1, n2)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            v = v / v_norm

            # Sjekk om linja er omtrent horisontal
            ridge_slope = math.degrees(math.atan2(abs(v[2]), math.sqrt(v[0]**2 + v[1]**2)))
            if ridge_slope > max_ridge_slope_deg:
                continue

            # Sjekk om normalene peker mot hverandre i xy-projeksjon
            cos_xy = float(np.dot(n1_xy, n2_xy) / (norm1_xy * norm2_xy))
            if cos_xy < min_cos:
                pairs.append((i, j))

    return pairs


# ---------------------------
# Segmentering av takflater
# ---------------------------

def segment_by_rgb(las):
    """
    Segmenterer punkter basert på unik RGB-kombinasjon.
    Returnerer:
        segments: liste med dict-er:
            {
              "segment_id": int,
              "mask": boolean numpy-array (len = n_points)
            }
    Hvis filen ikke har RGB, returneres én stor "segment" (hele taket).
    """
    dimensions = {d.lower() for d in las.point_format.dimension_names}

    has_rgb = {"red", "green", "blue"}.issubset(dimensions)
    n_points = len(las.x)

    if not has_rgb:
        # fallback: én flate med alle punkter
        return [{"segment_id": 0, "mask": np.ones(n_points, dtype=bool)}]

    r = np.asarray(las.red)
    g = np.asarray(las.green)
    b = np.asarray(las.blue)

    rgb = np.stack([r, g, b], axis=1)
    # Finn unike farger og en index per punkt
    unique_rgbs, inverse = np.unique(rgb, axis=0, return_inverse=True)

    segments = []
    for seg_id, rgb_val in enumerate(unique_rgbs):
        mask = (inverse == seg_id)
        # Dropp veldig små segmenter (kan være støy)
        if np.count_nonzero(mask) < 30:
            continue
        segments.append({"segment_id": seg_id, "mask": mask})

    # Hvis alt ble filtrert bort, fall tilbake til ett segment
    if not segments:
        segments = [{"segment_id": 0, "mask": np.ones(n_points, dtype=bool)}]

    return segments


# ---------------------------
# Taktype-heuristikk
# ---------------------------

def guess_roof_type(planes,
                    flat_slope_thresh_deg=5.0,
                    min_planes_for_complex=4):
    """
    Enkle regler for å gjette taktype ut fra planene og gavl-parene:
      - 1 plan, lav helning → Flat
      - 1 plan, ikke lav helning → Mono-pitch
      - 2+ plan, minst 1 gavl-par → Gabled / Complex_gabled
      - Mange plan, men ingen gavl-par → Hipped_or_pyramid
      - ellers → Unknown
    """
    n_planes = len(planes)
    if n_planes == 0:
        return "Unknown", 0, 0, 0.0

    slopes = [slope_from_plane(p) for p in planes]
    max_slope = max(slopes)
    mean_slope = sum(slopes) / len(slopes)

    if n_planes == 1:
        if max_slope < flat_slope_thresh_deg:
            return "Flat", n_planes, 0, mean_slope
        else:
            return "Mono_pitch", n_planes, 0, mean_slope

    # flere plan:
    gabled_pairs = find_gabled_pairs(planes)
    n_gabled = len(gabled_pairs)

    if all(s < flat_slope_thresh_deg for s in slopes):
        # Flere nesten-flate plan -> flatt/terrassert – vi kaller det Flat-ish
        return "Flat-ish", n_planes, n_gabled, mean_slope

    if n_gabled == 1:
        return "Gabled", n_planes, n_gabled, mean_slope
    elif n_gabled > 1:
        # Her kunne man skille T-element vs Cross ved å bygge en møne-graf
        return "Complex_gabled", n_planes, n_gabled, mean_slope

    # Ingen gabled-pairs, men en del plan → kanskje valmtak / pyramide
    if n_planes >= min_planes_for_complex and max_slope >= flat_slope_thresh_deg:
        return "Hipped_or_pyramid", n_planes, n_gabled, mean_slope

    return "Unknown", n_planes, n_gabled, mean_slope


# ---------------------------
# Hovedlogikk per fil
# ---------------------------

def process_roof_file(path: Path):
    """
    Leser en .laz-fil, segmenterer takflater, fiter plan,
    og gjetter en taktype.
    Returnerer en dict med nyttige verdier til CSV.
    """
    las = laspy.read(path)

    segments = segment_by_rgb(las)

    planes = []
    for seg in segments:
        mask = seg["mask"]
        x = las.x[mask]
        y = las.y[mask]
        z = las.z[mask]

        if len(x) < 3:
            continue

        plane = fit_plane_least_squares(x, y, z)
        planes.append(plane)

    roof_type_guess, n_planes, n_gabled, mean_slope = guess_roof_type(planes)

    roof_id = path.stem  # f.eks. "182486507"
    return {
        "roof_id": roof_id,
        "filename": str(path.name),
        "n_points_total": len(las.x),
        "n_segments": len(segments),
        "n_planes_used": n_planes,
        "n_gabled_pairs": n_gabled,
        "mean_slope_deg": round(mean_slope, 2),
        "roof_type_guess": roof_type_guess,
    }


def main():
    if len(sys.argv) < 2:
        print("Bruk: python guess_roof_types.py /sti/til/mappe_med_laz")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).expanduser().resolve()
    if not root_dir.is_dir():
        print(f"Fant ikke mappe: {root_dir}")
        sys.exit(1)

    laz_files = sorted(list(root_dir.glob("*.laz"))) + sorted(list(root_dir.glob("*.las")))

    if not laz_files:
        print(f"Ingen .laz/.las-filer funnet i {root_dir}")
        sys.exit(1)

    results = []
    for path in laz_files:
        print(f"Behandler {path.name} ...")
        try:
            res = process_roof_file(path)
        except Exception as e:
            print(f"  FEIL på {path.name}: {e}")
            res = {
                "roof_id": path.stem,
                "filename": path.name,
                "n_points_total": 0,
                "n_segments": 0,
                "n_planes_used": 0,
                "n_gabled_pairs": 0,
                "mean_slope_deg": 0.0,
                "roof_type_guess": f"Error: {type(e).__name__}",
            }
        results.append(res)

    out_csv = Path("roof_type_guesses.csv").resolve()
    fieldnames = [
        "roof_id",
        "filename",
        "n_points_total",
        "n_segments",
        "n_planes_used",
        "n_gabled_pairs",
        "mean_slope_deg",
        "roof_type_guess",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nFerdig!")
    print(f"Skrev taktype-gjetninger til: {out_csv}")
    print("Gå nå gjennom CSV-en og sjekk/juster taktypene manuelt der det trengs.")


if __name__ == "__main__":
    main()
