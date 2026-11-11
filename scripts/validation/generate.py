# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "h5netcdf==1.8.2", "h5py==3.16.0"]
# ///
"""Regenerate pinned external fixtures and R reference values, or verify their hashes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "test/external/sources.json"
REFERENCE = Path(__file__).with_name("reference.R")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, expected: str, path: Path) -> Path:
    if not path.exists():
        print(f"Downloading {path.name}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "mcmc-rs-validation"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Downloaded source checksum mismatch: {url}")
        path.write_bytes(payload)
    if digest(path) != expected:
        raise ValueError(f"Cached source checksum mismatch: {path}")
    return path


def run_r(rscript: str, env: dict, *args: str) -> None:
    subprocess.run([rscript, "--vanilla", str(REFERENCE), *args], env=env, check=True)


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    files = manifest["files"]
    for name, expected in files.items():
        if digest(output / name) != expected:
            raise ValueError(f"Generated fixture checksum mismatch: {name}")
    # The generation inputs are part of the reproducibility contract, too.
    for path, expected in manifest["generation_inputs"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Generation input changed without regeneration: {path}")
    print(f"Verified {len(files)} generated files and {len(manifest['generation_inputs'])} generation inputs.")


def generate(args: argparse.Namespace) -> None:
    import h5netcdf
    import numpy as np

    sources = json.loads(SOURCES.read_text())
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "licenses").mkdir(exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    ref = sources["reference"]
    package = fetch(ref["package_url"], ref["package_sha256"], cache / "posterior-1.7.1.tar.gz")
    library = cache / "reference-library"
    library.mkdir(exist_ok=True)
    env = dict(os.environ, R_LIBS_USER=str(library),
               MCMC_VALIDATION_R_LIBRARY=str(library), LC_ALL="C", TZ="UTC")
    stamp = library / "source-sha256"
    if not stamp.exists() or stamp.read_text().strip() != ref["package_sha256"]:
        subprocess.run([
            args.rscript, "--vanilla", "-e",
            'a <- commandArgs(TRUE); install.packages(a[1], repos = NULL, type = "source", lib = a[2])',
            str(package), str(library),
        ], env=env, check=True)
        # install.packages can report errors as warnings. Verify the private
        # installation rather than accidentally accepting a package elsewhere.
        subprocess.run([
            args.rscript, "--vanilla", "-e",
            'a <- commandArgs(TRUE); stopifnot(as.character(packageVersion("posterior", lib.loc = a[1])) == "1.7.1")',
            str(library),
        ], env=env, check=True)
        stamp.write_text(ref["package_sha256"] + "\n")

    fixture_metadata = []
    paths = []
    for fixture in sources["fixtures"]:
        path = output / (fixture["id"] + ".csv")
        metadata = dict(fixture)
        if fixture["type"] == "posterior":
            mapping = [item for pair in fixture["parameters"] for item in pair]
            run_r(args.rscript, env, "export", fixture["example"], str(path), *mapping)
            metadata["source_package_sha256"] = ref["package_sha256"]
        else:
            original = fetch(fixture["url"], fixture["sha256"], cache / fixture["cache_file"])
            with h5netcdf.File(original, "r") as netcdf:
                posterior = netcdf.groups["posterior"]
                columns = []
                for alias, variable, indices in fixture["parameters"]:
                    var = posterior.variables[variable]
                    if var.dimensions[:2] != ("chain", "draw"):
                        raise ValueError(f"Unexpected source dimension order: {fixture['id']}/{variable}")
                    values = np.asarray(var[(slice(None), slice(None), *indices)], dtype=np.float64)
                    if values.ndim != 2 or not np.all(np.isfinite(values)):
                        raise ValueError(f"Invalid source draws: {fixture['id']}/{variable}")
                    columns.append(values)
                m, n = columns[0].shape
                if any(values.shape != (m, n) for values in columns):
                    raise ValueError("Source variables have different chain/draw dimensions")
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(["chain", "draw"] + [p[0] for p in fixture["parameters"]])
                    for chain in range(m):
                        for draw in range(n):
                            writer.writerow([chain + 1, draw + 1] + [
                                format(values[chain, draw], ".17g") for values in columns
                            ])
                metadata["sampler_attributes"] = {
                    key: str(value) for key, value in posterior.attrs.items()
                }
        payload = path.read_text()
        path.write_text(
            "# Modified excerpt: selected variables, original chain/draw order; provenance and licenses in sources.json and NOTICE.\n"
            + payload
        )
        with path.open() as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
        metadata["chains"] = max(int(row["chain"]) for row in rows)
        metadata["draws_per_chain"] = max(int(row["draw"]) for row in rows)
        if len(rows) != metadata["chains"] * metadata["draws_per_chain"]:
            raise ValueError(f"Incomplete chain rows: {path}")
        metadata["extraction"] = "Selected named variables/elements; all stored posterior draws in original chain order; no thinning or resampling. CSV aliases and zero-based element indices are listed in parameters."
        fixture_metadata.append(metadata)
        paths.append(path)
        print(f"{fixture['id']}: {metadata['chains']} chains x {metadata['draws_per_chain']} draws", flush=True)

    expected = output / "expected.csv"
    run_r(args.rscript, env, "reference", str(expected), *(str(path) for path in paths))
    for license in sources["licenses"]:
        original = fetch(license["url"], license["sha256"], cache / license["file"])
        shutil.copyfile(original, output / "licenses" / license["file"])

    notice = ROOT / "test/external/NOTICE"
    if output / "NOTICE" != notice:
        shutil.copyfile(notice, output / "NOTICE")

    inputs = [SOURCES, REFERENCE, Path(__file__).resolve(), Path(__file__).with_name("environment.yml"), notice]
    generated = paths + [expected, output / "semantics.csv", output / "NOTICE"] + sorted((output / "licenses").glob("*.txt"))
    manifest = {
        "schema_version": 1,
        "reference": ref,
        "conversion_versions": {"numpy": np.__version__, "h5netcdf": h5netcdf.__version__},
        "fixtures": fixture_metadata,
        "files": {str(path.relative_to(output)): digest(path) for path in generated},
        "generation_inputs": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify checked-in files without R, downloads, or Python dependencies")
    parser.add_argument("--output", type=Path, default=ROOT / "test/external")
    parser.add_argument("--cache", type=Path, default=ROOT / "target/external-validation")
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    if args.check:
        verify(args.output)
    else:
        generate(args)


if __name__ == "__main__":
    main()
