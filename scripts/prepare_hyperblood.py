#!/usr/bin/env python3
"""Turn the HyperBlood archive into inputs hsi_detect can consume.

Produces, for each scene:

  * a cleaned ENVI cube - 113 bands, BSQ, float32
  * a raw uint8 ground-truth mask, raster order, class index per pixel

and, once over the whole set:

  * a target spectral library as CSV, one mean spectrum per class

Two pieces of the raw data have to be dealt with before any of it is useful.

The cubes ship with 128 bands, of which 15 are noise: the first five and the
last seven sit at the edges of the sensor's response where there is almost no
signal, and indices 48-50 are a known artefact. The published band list is used
verbatim, which is also what makes these spectra comparable to results anyone
else reports on this dataset.

And the classes are chosen to be hard for the right reason. Blood sits beside
ketchup, tomato concentrate, beetroot juice and two paints - all of them red,
none of them separable by colour. That is the case the spectral angle exists
for, so the library deliberately keeps the lookalikes as targets of their own
rather than folding them into the background.

    python3 scripts/prepare_hyperblood.py [--data-dir data] [--scene F_1 ...]
"""

import argparse
import csv
import pathlib
import sys

import numpy as np

# Published cleaning for this dataset: sensor-edge bands plus a known artefact
# around index 48-50. 128 - 15 = 113.
NOISY_BANDS = np.array(
    [0, 1, 2, 3, 4, 48, 49, 50, 121, 122, 123, 124, 125, 126, 127]
)

CLASS_NAMES = {
    0: "background",
    1: "blood",
    2: "ketchup",
    3: "artificial_blood",
    4: "beetroot_juice",
    5: "poster_paint",
    6: "tomato_concentrate",
    7: "acrylic_paint",
    8: "uncertain_blood",
}


def parse_envi_header(path):
    """Minimal ENVI header parse. Scans character-wise because the wavelength
    list wraps across lines and ';' comment lines must not become fields."""
    text = path.read_text(errors="replace")
    fields, pos = {}, 0
    while True:
        eq = text.find("=", pos)
        if eq < 0:
            break
        line_start = text.rfind("\n", 0, eq)
        key = text[line_start + 1 : eq].strip().lower()
        rest = text[eq + 1 :].lstrip(" \t")
        offset = eq + 1 + (len(text) - eq - 1 - len(rest))
        if rest.startswith("{"):
            close = text.find("}", offset)
            value, pos = text[offset + 1 : close], close + 1
        else:
            eol = text.find("\n", offset)
            eol = len(text) if eol < 0 else eol
            value, pos = text[offset:eol], eol + 1
        if key and not key.startswith(";"):
            fields[key] = value.strip()
    return fields


def read_cube(hdr_path):
    """Read an ENVI cube as (bands, height, width) float32, plus wavelengths."""
    fields = parse_envi_header(hdr_path)
    width = int(fields["samples"])
    height = int(fields["lines"])
    bands = int(fields["bands"])
    interleave = fields.get("interleave", "bsq").lower()
    if int(fields.get("data type", 4)) != 4:
        raise SystemExit(f"{hdr_path}: expected float32 (data type 4)")

    data_path = hdr_path.with_suffix(".float")
    if not data_path.exists():
        for suffix in ("", ".dat", ".img", ".bin"):
            candidate = hdr_path.with_suffix(suffix)
            if candidate.exists() and candidate != hdr_path:
                data_path = candidate
                break
    raw = np.fromfile(data_path, dtype="<f4")
    if raw.size != width * height * bands:
        raise SystemExit(
            f"{data_path}: {raw.size} samples, expected {width * height * bands}"
        )

    if interleave == "bil":
        cube = raw.reshape(height, bands, width).transpose(1, 0, 2)
    elif interleave == "bip":
        cube = raw.reshape(height, width, bands).transpose(2, 0, 1)
    else:
        cube = raw.reshape(bands, height, width)

    wavelengths = None
    if "wavelength" in fields:
        try:
            wavelengths = np.array(
                [float(v) for v in fields["wavelength"].replace("\n", " ").split(",")]
            )
        except ValueError:
            wavelengths = None
    return np.ascontiguousarray(cube, dtype=np.float32), wavelengths


def write_envi(path_stem, cube, wavelengths):
    """Write (bands, height, width) float32 as a BSQ ENVI pair."""
    bands, height, width = cube.shape
    cube.astype("<f4").tofile(path_stem.with_suffix(".float"))
    lines = [
        "ENVI",
        ";cleaned HyperBlood cube: noisy bands removed, converted to BSQ",
        ";source: https://doi.org/10.5281/zenodo.3984905 (CC-BY-4.0)",
        "description = {}",
        f"samples = {width}",
        f"lines = {height}",
        f"bands = {bands}",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
    ]
    if wavelengths is not None:
        joined = " , ".join(f"{w:.4f}" for w in wavelengths)
        lines.append("wavelength = { " + joined + " }")
    path_stem.with_suffix(".hdr").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", type=pathlib.Path)
    parser.add_argument(
        "--scene", action="append", help="scene to process; repeatable, default all"
    )
    parser.add_argument(
        "--keep-noisy-bands",
        action="store_true",
        help="skip band cleaning and keep all 128 bands",
    )
    args = parser.parse_args()

    root = args.data_dir / "HyperBlood"
    if not root.is_dir():
        raise SystemExit(f"{root} not found - run scripts/fetch_hyperblood.sh first")

    out_dir = args.data_dir / "hyperblood_prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = sorted((root / "data").glob("*.hdr"))
    if args.scene:
        wanted = set(args.scene)
        headers = [h for h in headers if h.stem in wanted]
    if not headers:
        raise SystemExit("no scenes found")

    # Accumulated per class across every scene, so one library covers them all
    # rather than overfitting a signature to a single lighting condition.
    totals, counts, wavelengths_out = {}, {}, None

    for hdr in headers:
        scene = hdr.stem
        cube, wavelengths = read_cube(hdr)

        anno_path = root / "anno" / f"{scene}.npz"
        if not anno_path.exists():
            print(f"{scene}: no annotation, skipping", file=sys.stderr)
            continue
        gt = np.load(anno_path)["gt"]
        if gt.shape != cube.shape[1:]:
            print(
                f"{scene}: annotation {gt.shape} does not match cube "
                f"{cube.shape[1:]}, skipping",
                file=sys.stderr,
            )
            continue

        if not args.keep_noisy_bands:
            keep = np.setdiff1d(np.arange(cube.shape[0]), NOISY_BANDS)
            cube = cube[keep]
            if wavelengths is not None and len(wavelengths) == len(keep) + len(NOISY_BANDS):
                wavelengths = wavelengths[keep]
        if wavelengths_out is None:
            wavelengths_out = wavelengths

        write_envi(out_dir / scene, cube, wavelengths)

        # Labels above 8 are annotation bookkeeping rather than materials, so
        # they are written into the mask as-is but never used as a signature.
        mask = np.where(gt <= 8, gt, 255).astype(np.uint8)
        (out_dir / f"{scene}_gt.u8").write_bytes(mask.tobytes())

        present = []
        for label in np.unique(gt):
            if label == 0 or label > 8:
                continue
            selected = cube[:, gt == label]
            totals[int(label)] = totals.get(int(label), 0.0) + selected.sum(axis=1)
            counts[int(label)] = counts.get(int(label), 0) + selected.shape[1]
            present.append(CLASS_NAMES.get(int(label), str(label)))
        print(
            f"{scene}: {cube.shape[0]} bands, {cube.shape[1]}x{cube.shape[2]}, "
            f"classes: {', '.join(present)}"
        )

    if not totals:
        raise SystemExit("no annotated pixels found")

    library_path = args.data_dir / "hyperblood_targets.csv"
    with library_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        handle.write("# HyperBlood mean class spectra\n")
        handle.write("# https://doi.org/10.5281/zenodo.3984905 (CC-BY-4.0)\n")
        handle.write(
            "# blood sits beside four red lookalikes on purpose - that is the\n"
            "# case colour cannot separate and the spectral angle can\n"
        )
        if wavelengths_out is not None:
            writer.writerow(["wavelength"] + [f"{w:.4f}" for w in wavelengths_out])
        for label in sorted(totals):
            mean = totals[label] / counts[label]
            writer.writerow(
                [CLASS_NAMES.get(label, str(label))] + [f"{v:.6f}" for v in mean]
            )
            print(f"  {CLASS_NAMES.get(label, label):>20}: {counts[label]:7d} pixels")

    print(f"\nwrote {library_path}")
    print(f"wrote cleaned cubes and masks to {out_dir}")


if __name__ == "__main__":
    main()
