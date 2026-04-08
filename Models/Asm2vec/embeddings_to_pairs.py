#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Convert function embeddings into pairwise similarity CSVs.                #
#                                                                            #
##############################################################################


import argparse
import csv
import logging
import os

import numpy as np


log = logging.getLogger("embeddings_to_pairs")


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


def load_embeddings(embeddings_path):
    """Load embeddings.csv into memory keyed by idb_path:fva."""
    embeddings = dict()

    log.info("[*] Loading embeddings from %s", embeddings_path)
    with open(embeddings_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        required = {"idb_path", "fva", "embeddings"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Expected embeddings CSV columns {}, found {}".format(
                    sorted(required), reader.fieldnames))

        for row_idx, row in enumerate(reader, start=2):
            func_key = "{}:{}".format(row["idb_path"], row["fva"])
            vector = np.fromstring(row["embeddings"], sep=";", dtype=np.float32)
            if vector.size == 0:
                raise ValueError(
                    "Empty embedding at row {} in {}".format(
                        row_idx, embeddings_path))
            embeddings[func_key] = vector

    log.info("\tLoaded %d embeddings", len(embeddings))
    return embeddings


def get_embedding(embeddings, idb_path, fva, pair_path, row_idx, side):
    """Fetch one function embedding and fail loudly if it is missing."""
    func_key = "{}:{}".format(idb_path, fva)
    if func_key not in embeddings:
        raise KeyError(
            "Missing embedding for {} in {} at row {} ({})".format(
                func_key, pair_path, row_idx, side))
    return embeddings[func_key]


def cosine_similarity(emb_1, emb_2):
    """Compute cosine similarity between two numpy vectors."""
    denom = np.linalg.norm(emb_1) * np.linalg.norm(emb_2)
    if denom == 0:
        return 0.0
    return float(np.dot(emb_1, emb_2) / denom)


def score_pair_csv(embeddings, pair_path, output_path):
    """Append a sim column to one pair CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
                emb_1 = get_embedding(
                    embeddings,
                    row["idb_path_1"],
                    row["fva_1"],
                    pair_path,
                    row_idx,
                    "left",
                )
                emb_2 = get_embedding(
                    embeddings,
                    row["idb_path_2"],
                    row["fva_2"],
                    pair_path,
                    row_idx,
                    "right",
                )
                row["sim"] = cosine_similarity(emb_1, emb_2)
                writer.writerow(row)
                num_rows += 1

    log.info("[*] Wrote %d scored pairs to %s", num_rows, output_path)


def write_pair_similarities(embeddings, pairs_dir, outputdir):
    """Score all pair CSVs under pairs_dir and mirror them into outputdir."""
    csv_paths = list(iter_pair_csv_paths(pairs_dir))
    if not csv_paths:
        raise ValueError("No pair CSVs found in {}".format(pairs_dir))

    log.info("[*] Scoring %d pair CSVs from %s", len(csv_paths), pairs_dir)
    for pair_path in csv_paths:
        rel_path = os.path.relpath(pair_path, pairs_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name, ext = os.path.splitext(os.path.basename(rel_path))

        output_subdir = outputdir
        if rel_dir and rel_dir != ".":
            output_subdir = os.path.join(outputdir, rel_dir)

        output_path = os.path.join(
            output_subdir, "{}_sim{}".format(base_name, ext))
        score_pair_csv(embeddings, pair_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        prog="embeddings_to_pairs.py",
        description="Convert embeddings.csv and pair CSVs into *_sim.csv files",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "-e", "--embeddings", required=True, help="Path to embeddings.csv")
    parser.add_argument(
        "-p", "--pairs-dir", required=True,
        help="Directory containing pair CSVs")
    parser.add_argument(
        "-o", "--outputdir", required=True,
        help="Directory where *_sim.csv files will be written")

    args = parser.parse_args()
    set_logger(args.debug)

    if not os.path.isfile(args.embeddings):
        raise ValueError("Embeddings CSV not found: {}".format(args.embeddings))
    if not os.path.isdir(args.pairs_dir):
        raise ValueError("Pairs directory not found: {}".format(args.pairs_dir))

    os.makedirs(args.outputdir, exist_ok=True)

    embeddings = load_embeddings(args.embeddings)
    write_pair_similarities(embeddings, args.pairs_dir, args.outputdir)


if __name__ == "__main__":
    main()
