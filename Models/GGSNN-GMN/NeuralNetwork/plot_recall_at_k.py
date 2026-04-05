#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot a high-DPI Recall@K curve for function similarity search results.

The ranking CSVs produced by this repository store one positive candidate in
`pos_rank*_sim.csv` and the corresponding negative candidates in
`neg_rank*_sim.csv`. In the released datasets, the files are ordered so that
the i-th positive row matches the block of negatives:

    neg_rows[i * negatives_per_positive : (i + 1) * negatives_per_positive]

This script streams the CSV files instead of loading everything in memory,
which keeps it practical for larger ranking files.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from textwrap import fill
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker


QUERY_COLUMNS = ("idb_path_1", "fva_1", "func_name_1", "db_type")
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "Dataset-1_testing"
DEFAULT_DPI = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot Recall@K from pos/neg ranking CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing pos_rank*_sim.csv and neg_rank*_sim.csv.",
    )
    parser.add_argument(
        "--pos-csv",
        type=Path,
        default=None,
        help="Explicit path to the positive ranking CSV.",
    )
    parser.add_argument(
        "--neg-csv",
        type=Path,
        default=None,
        help="Explicit path to the negative ranking CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <input-dir>/recall_at_k.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output CSV path for the computed curve. Default: <input-dir>/recall_at_k.csv",
    )
    parser.add_argument(
        "--score-column",
        default="sim",
        help="Column name that stores the similarity score.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=None,
        help="Fixed number of negative candidates for each positive candidate. If omitted, it is inferred from the file sizes.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Maximum K shown in the curve. Default: min(100, candidates_per_query).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Figure DPI. Default: 400",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title for the plot.",
    )
    parser.add_argument(
        "--group-column",
        default="db_type",
        help="Column used to split the Recall@K curves into task-specific subplots. Default: db_type",
    )
    parser.add_argument(
        "--hide-overall",
        action="store_true",
        help="Do not add an overall subplot when multiple task groups are present.",
    )
    parser.add_argument(
        "--no-query-check",
        action="store_true",
        help="Skip validation that each negative block belongs to the same query as its positive row.",
    )
    return parser.parse_args()


def resolve_csv(input_dir: Path, explicit_path: Path | None, pattern: str, label: str) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()

    matches = sorted(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label} under {input_dir} with pattern {pattern!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {label} files under {input_dir}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def get_query_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(row.get(column, "") for column in QUERY_COLUMNS if column in row)


def infer_negatives_per_positive(
    pos_csv: Path,
    neg_csv: Path,
    user_value: int | None,
) -> Tuple[int, int, int]:
    num_pos = count_csv_rows(pos_csv)
    num_neg = count_csv_rows(neg_csv)

    if num_pos <= 0:
        raise ValueError(f"No positive rows found in {pos_csv}.")
    if num_neg <= 0:
        raise ValueError(f"No negative rows found in {neg_csv}.")

    if user_value is not None:
        if user_value <= 0:
            raise ValueError("--negatives-per-positive must be > 0.")
        expected_neg = num_pos * user_value
        if expected_neg != num_neg:
            raise ValueError(
                f"Expected {expected_neg} negative rows from {num_pos} positives "
                f"and {user_value} negatives per positive, but found {num_neg}."
            )
        return user_value, num_pos, num_neg

    if num_neg % num_pos != 0:
        raise ValueError(
            f"Cannot infer a fixed negatives-per-positive ratio: "
            f"{num_neg} negative rows is not divisible by {num_pos} positive rows."
        )

    return num_neg // num_pos, num_pos, num_neg


def compute_rank_histograms(
    pos_csv: Path,
    neg_csv: Path,
    score_column: str,
    negatives_per_positive: int,
    validate_query_alignment: bool,
    group_column: str,
) -> Tuple[List[int], Dict[str, List[int]]]:
    max_rank = negatives_per_positive + 1
    overall_histogram = [0] * (max_rank + 1)
    grouped_histograms: Dict[str, List[int]] = {}

    with pos_csv.open("r", newline="") as pos_handle, neg_csv.open("r", newline="") as neg_handle:
        pos_reader = csv.DictReader(pos_handle)
        neg_reader = csv.DictReader(neg_handle)

        for episode_index, pos_row in enumerate(pos_reader):
            try:
                positive_score = float(pos_row[score_column])
            except KeyError as exc:
                raise KeyError(
                    f"Column {score_column!r} was not found in {pos_csv}."
                ) from exc

            positive_query = get_query_key(pos_row)
            rank = 1

            for _ in range(negatives_per_positive):
                neg_row = next(neg_reader, None)
                if neg_row is None:
                    raise ValueError(
                        f"Negative CSV ended early while processing episode {episode_index}."
                    )
                if validate_query_alignment and positive_query:
                    negative_query = get_query_key(neg_row)
                    if negative_query != positive_query:
                        raise ValueError(
                            "The ordering assumption is violated between positive "
                            f"and negative CSV files at episode {episode_index}.\n"
                            f"Positive query: {positive_query}\n"
                            f"Negative query: {negative_query}"
                        )
                try:
                    negative_score = float(neg_row[score_column])
                except KeyError as exc:
                    raise KeyError(
                        f"Column {score_column!r} was not found in {neg_csv}."
                    ) from exc

                # Larger similarity is better. Ties are treated in favor of the
                # positive candidate, which keeps the rank definition stable.
                if negative_score > positive_score:
                    rank += 1

            task_name = "ALL"
            if group_column and group_column in pos_row:
                task_name = pos_row.get(group_column, "") or "UNSPECIFIED"
            if task_name not in grouped_histograms:
                grouped_histograms[task_name] = [0] * (max_rank + 1)

            overall_histogram[rank] += 1
            grouped_histograms[task_name][rank] += 1

        trailing_negative = next(neg_reader, None)
        if trailing_negative is not None:
            raise ValueError(
                "Negative CSV still contains unread rows after all positives were processed."
            )

    return overall_histogram, grouped_histograms


def histogram_to_ranks(rank_histogram: Sequence[int]) -> List[int]:
    ranks: List[int] = []
    for rank, count in enumerate(rank_histogram):
        if rank == 0 or count == 0:
            continue
        ranks.extend([rank] * count)
    return ranks


def compute_recall_curve(
    rank_histogram: Sequence[int],
    max_k: int,
) -> Tuple[List[int], List[float]]:
    total = sum(rank_histogram)
    if total <= 0:
        raise ValueError("Rank histogram is empty.")

    recalls: List[float] = []
    ks = list(range(1, max_k + 1))
    cumulative_hits = 0
    for k in ks:
        cumulative_hits += rank_histogram[k]
        recalls.append(cumulative_hits / total)
    return ks, recalls


def save_curve_csv(
    path: Path,
    histogram_map: Dict[str, Sequence[int]],
    max_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "k", "recall", "episodes"])
        for task_name, histogram in histogram_map.items():
            ks, recalls = compute_recall_curve(histogram, max_k=max_k)
            episodes = sum(histogram)
            for k, recall in zip(ks, recalls):
                writer.writerow([task_name, k, f"{recall:.10f}", episodes])


def recall_at(rank_histogram: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    capped_k = min(k, len(rank_histogram) - 1)
    total = sum(rank_histogram)
    hits = sum(rank_histogram[1 : capped_k + 1])
    return hits / total


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def plot_curve_grid(
    histogram_map: Dict[str, Sequence[int]],
    output_path: Path,
    title: str,
    subtitle: str,
    max_k: int,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    num_plots = len(histogram_map)
    num_cols = 1 if num_plots == 1 else 2 if num_plots <= 4 else 3
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6.7 * num_cols, 4.7 * num_rows),
        dpi=dpi,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    fig.patch.set_facecolor("#f4f7fb")
    fig.subplots_adjust(top=0.84, left=0.06, right=0.985, bottom=0.08, hspace=0.28, wspace=0.18)

    palette = [
        ("#0f172a", "#94a3b8"),
        ("#0f766e", "#2dd4bf"),
        ("#1d4ed8", "#60a5fa"),
        ("#be123c", "#fb7185"),
        ("#6d28d9", "#a78bfa"),
        ("#c2410c", "#fdba74"),
    ]
    point_color = "#111827"
    grid_color = "#d7e3f0"

    wrapped_title = fill(title, width=38)
    wrapped_subtitle = fill(subtitle, width=96)
    fig.suptitle(wrapped_title, fontsize=16.5, color="#0f172a", y=0.992, fontweight="bold")
    fig.text(0.5, 0.958, wrapped_subtitle, fontsize=9.6, color="#475569", ha="center")

    for panel_index, (task_name, rank_histogram) in enumerate(histogram_map.items()):
        ax = axes_list[panel_index]
        line_color, fill_color = palette[panel_index % len(palette)]
        ks, recalls = compute_recall_curve(rank_histogram, max_k=max_k)
        anchor_candidates = [1, 5, 10, 20, 50, max_k]
        anchor_ks = []
        for candidate in anchor_candidates:
            if 1 <= candidate <= max_k and candidate not in anchor_ks:
                anchor_ks.append(candidate)
        anchor_recalls = [recalls[k - 1] for k in anchor_ks]

        ax.set_facecolor("#fbfdff")
        ax.step(ks, recalls, where="post", color=line_color, linewidth=2.7)
        ax.fill_between(ks, recalls, step="post", color=fill_color, alpha=0.16)
        ax.scatter(anchor_ks, anchor_recalls, s=24, color=point_color, zorder=4)

        ax.set_xlim(1, max_k)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.set_xticks(anchor_ks)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(True, which="major", axis="both", color=grid_color, linewidth=0.85, alpha=0.95)
        ax.grid(True, which="minor", axis="y", color=grid_color, linewidth=0.5, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9fb3c8")
        ax.spines["bottom"].set_color("#9fb3c8")
        ax.tick_params(axis="both", labelsize=10, colors="#334155")

        panel_title = "Overall" if task_name == "ALL" else f"Task: {task_name}"
        ax.set_title(panel_title, fontsize=13, color="#0f172a", pad=10)
        ax.set_xlabel("K", fontsize=11.5, color="#0f172a")
        ax.set_ylabel("Recall", fontsize=11.5, color="#0f172a")

        ranks = histogram_to_ranks(rank_histogram)
        summary_lines = [
            f"Episodes: {len(ranks):,}",
            f"Median rank: {median(ranks):.0f}",
            f"R@1: {format_percent(recall_at(rank_histogram, 1))}",
            f"R@5: {format_percent(recall_at(rank_histogram, 5))}",
            f"R@10: {format_percent(recall_at(rank_histogram, 10))}",
        ]
        ax.text(
            0.985,
            0.02,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.6,
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.36",
                "facecolor": "#ffffff",
                "edgecolor": "#d7e3f0",
                "alpha": 0.96,
            },
        )

    for ax in axes_list[len(histogram_map) :]:
        ax.set_visible(False)

    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    pos_csv = resolve_csv(input_dir, args.pos_csv, "pos_rank*_sim.csv", "positive ranking CSV")
    neg_csv = resolve_csv(input_dir, args.neg_csv, "neg_rank*_sim.csv", "negative ranking CSV")

    output_path = args.output.resolve() if args.output else input_dir / "recall_at_k.png"
    csv_output_path = (
        args.csv_output.resolve() if args.csv_output else input_dir / "recall_at_k.csv"
    )

    negatives_per_positive, num_pos, num_neg = infer_negatives_per_positive(
        pos_csv=pos_csv,
        neg_csv=neg_csv,
        user_value=args.negatives_per_positive,
    )
    candidates_per_query = negatives_per_positive + 1

    max_k = args.max_k if args.max_k is not None else min(100, candidates_per_query)
    if max_k <= 0:
        raise ValueError("--max-k must be > 0.")
    max_k = min(max_k, candidates_per_query)

    overall_histogram, grouped_histograms = compute_rank_histograms(
        pos_csv=pos_csv,
        neg_csv=neg_csv,
        score_column=args.score_column,
        negatives_per_positive=negatives_per_positive,
        validate_query_alignment=not args.no_query_check,
        group_column=args.group_column,
    )

    histogram_map: Dict[str, Sequence[int]] = {}
    if len(grouped_histograms) > 1 and not args.hide_overall:
        histogram_map["ALL"] = overall_histogram
    histogram_map.update(grouped_histograms)

    title = args.title or f"Recall@K on {input_dir.name}"
    subtitle = (
        f"1 positive + {negatives_per_positive} negatives per query, "
        f"grouped by '{args.group_column}', score='{args.score_column}'"
    )

    save_curve_csv(csv_output_path, histogram_map, max_k=max_k)
    plot_curve_grid(
        histogram_map=histogram_map,
        output_path=output_path,
        title=title,
        subtitle=subtitle,
        max_k=max_k,
        dpi=args.dpi,
    )

    print(f"Positive CSV: {pos_csv}")
    print(f"Negative CSV: {neg_csv}")
    print(f"Episodes: {num_pos:,}")
    print(f"Negative rows: {num_neg:,}")
    print(f"Negatives per positive: {negatives_per_positive}")
    print(f"Candidates per query: {candidates_per_query}")
    print(f"Task groups: {', '.join(grouped_histograms.keys())}")
    print(f"Saved figure: {output_path}")
    print(f"Saved curve CSV: {csv_output_path}")
    for task_name, rank_histogram in histogram_map.items():
        summary = {
            1: recall_at(rank_histogram, 1),
            5: recall_at(rank_histogram, 5),
            10: recall_at(rank_histogram, 10),
        }
        label = "Overall" if task_name == "ALL" else task_name
        print(
            f"{label}: "
            + ", ".join(f"Recall@{k}={format_percent(value)}" for k, value in summary.items())
        )


if __name__ == "__main__":
    main()
