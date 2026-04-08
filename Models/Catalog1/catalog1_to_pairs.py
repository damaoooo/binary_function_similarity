#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Convert one Catalog1 feature CSV plus an existing pairs directory into    #
#  repository-style *_sim.csv files.                                         #
#                                                                            #
##############################################################################

import argparse
import csv
import logging
import os


log = logging.getLogger("catalog1_to_pairs")


def set_logger(debug):
    """Configure a simple stderr logger."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s:: %(message)s",
        datefmt="%H:%M:%S",
    )


def iter_pair_csv_paths(pairs_dir):
    """Yield pair CSVs in a deterministic order."""
    csv_paths = []
    for root, _, files in os.walk(pairs_dir):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            if fname.endswith("_sim.csv"):
                continue
            csv_paths.append(os.path.join(root, fname))

    for csv_path in sorted(csv_paths, key=lambda p: os.path.relpath(p, pairs_dir)):
        yield csv_path


def parse_catalog_hashes(raw_value):
    """Parse the Catalog1 signature string into a set of tokens."""
    if not raw_value:
        return set()
    if raw_value == "min_function_size_error":
        return set()
    return set(token for token in raw_value.split(";") if token)


def load_catalog_signatures(catalog_csv):
    """Load one Catalog1 CSV keyed by (idb_path, fva)."""
    signatures = {}

    log.info("[*] Loading Catalog1 signatures from %s", catalog_csv)
    with open(catalog_csv, newline="") as f_in:
        reader = csv.DictReader(f_in)
        required = {"path", "address", "catalog_hash_list"}
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                "Missing required Catalog1 columns {} in {}".format(
                    missing, catalog_csv))

        for row_idx, row in enumerate(reader, start=2):
            key = (row["path"], row["address"].lower())
            signatures[key] = parse_catalog_hashes(row["catalog_hash_list"])

    log.info("\tLoaded %d function signatures", len(signatures))
    return signatures


def get_signature(signatures, idb_path, fva, pair_path, row_idx, side):
    """Fetch one Catalog1 signature or fail loudly if missing."""
    key = (idb_path, fva.lower())
    if key not in signatures:
        raise KeyError(
            "Missing Catalog1 signature for {}:{} in {} at row {} ({})".format(
                idb_path, fva, pair_path, row_idx, side))
    return signatures[key]


def jaccard_similarity(sig_left, sig_right):
    """Compute Jaccard similarity between two signature sets."""
    union = sig_left | sig_right
    if not union:
        return 0.0
    return float(len(sig_left & sig_right) / len(union))


def score_pair_csv(signatures, pair_path, output_path):
    """Append a sim column to one pair CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(pair_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if "sim" not in fieldnames:
            fieldnames.append("sim")

        with open(output_path, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            num_rows = 0
            for row_idx, row in enumerate(reader, start=2):
                sig_left = get_signature(
                    signatures,
                    row["idb_path_1"],
                    row["fva_1"],
                    pair_path,
                    row_idx,
                    "left",
                )
                sig_right = get_signature(
                    signatures,
                    row["idb_path_2"],
                    row["fva_2"],
                    pair_path,
                    row_idx,
                    "right",
                )
                row["sim"] = jaccard_similarity(sig_left, sig_right)
                writer.writerow(row)
                num_rows += 1

    log.info("[*] Wrote %d scored pairs to %s", num_rows, output_path)


def write_pair_similarities(signatures, pairs_dir, output_dir):
    """Score every pair CSV under pairs_dir and mirror them into output_dir."""
    csv_paths = list(iter_pair_csv_paths(pairs_dir))
    if not csv_paths:
        raise ValueError("No pair CSVs found in {}".format(pairs_dir))

    log.info("[*] Scoring %d pair CSVs from %s", len(csv_paths), pairs_dir)
    for pair_path in csv_paths:
        rel_path = os.path.relpath(pair_path, pairs_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name, ext = os.path.splitext(os.path.basename(rel_path))

        target_dir = output_dir
        if rel_dir and rel_dir != ".":
            target_dir = os.path.join(output_dir, rel_dir)

        output_path = os.path.join(target_dir, "{}_sim{}".format(base_name, ext))
        score_pair_csv(signatures, pair_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        prog="catalog1_to_pairs.py",
        description="Convert one Catalog1 CSV and an existing pairs directory "
                    "into *_sim.csv files",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "-c", "--catalog-csv", required=True, help="Path to one Catalog1 CSV")
    parser.add_argument(
        "-p", "--pairs-dir", required=True,
        help="Directory containing existing pair CSVs")
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory where *_sim.csv files will be written")

    args = parser.parse_args()
    set_logger(args.debug)

    if not os.path.isfile(args.catalog_csv):
        raise ValueError("Catalog1 CSV not found: {}".format(args.catalog_csv))
    if not os.path.isdir(args.pairs_dir):
        raise ValueError("Pairs directory not found: {}".format(args.pairs_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    signatures = load_catalog_signatures(args.catalog_csv)
    write_pair_similarities(signatures, args.pairs_dir, args.output_dir)


if __name__ == "__main__":
    main()
