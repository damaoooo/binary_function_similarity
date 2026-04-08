#!/usr/bin/env python3

from __future__ import annotations

##############################################################################
#                                                                            #
#  Code for the USENIX Security '22 paper:                                   #
#  How Machine Learning Is Solving the Binary Function Similarity Problem.   #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#                                                                            #
#  convert_fss_results.py - Convert raw FunctionSimSearch CSVs into          #
#  pairwise similarity CSVs without using the notebook.                      #
#                                                                            #
##############################################################################

import concurrent.futures
import csv
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple

import click

AUTO_MAX_JOBS = 16
TARGET_FSS_CSV_NAME = 'IMM:0.00_MNEM:0.00_GRAPH:1.00.csv'

FSS_REQUIRED_COLUMNS = {'path', 'address', 'hashes0', 'hashes1'}
PAIR_REQUIRED_COLUMNS = {'idb_path_1', 'fva_1', 'idb_path_2', 'fva_2'}

_FSS_CACHE: Dict[str, Dict[Tuple[str, str], Tuple[int, int]]] = {}


@dataclass(frozen=True)
class PairSpec:
    csv_path: str
    output_rel_path: str
    row_count: int


@dataclass(frozen=True)
class ConversionTask:
    fss_csv: str
    pair_csv: str
    output_csv: str
    row_count: int


def get_parallel_jobs(jobs: int, task_count: int) -> int:
    """Return the effective number of worker processes."""
    if task_count <= 0:
        return 1

    if jobs == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, cpu_count, AUTO_MAX_JOBS))

    return max(1, min(jobs, task_count))


def collect_fss_csvs(fss_input: Path) -> list[Path]:
    """Return the raw FunctionSimSearch CSVs to convert."""
    if not fss_input.exists():
        raise click.ClickException(f'Input path does not exist: {fss_input}')

    if fss_input.is_file():
        if fss_input.suffix.lower() != '.csv':
            raise click.ClickException(
                f'Expected a CSV file, got: {fss_input}')
        return [fss_input]

    target_csv_path = fss_input / TARGET_FSS_CSV_NAME
    if target_csv_path.is_file():
        return [target_csv_path]

    csv_paths = sorted(
        path.name for path in fss_input.iterdir()
        if path.is_file() and path.suffix.lower() == '.csv')
    if not csv_paths:
        raise click.ClickException(
            f'No CSV files found in input directory: {fss_input}')
    raise click.ClickException(
        f'Could not find {TARGET_FSS_CSV_NAME} in {fss_input}. '
        f'Available CSVs: {", ".join(csv_paths)}')


def count_csv_rows(csv_path: Path) -> int:
    """Count non-empty data rows in a CSV file."""
    with csv_path.open(newline='') as f_in:
        reader = csv.reader(f_in)
        try:
            next(reader)
        except StopIteration:
            return 0

        row_count = 0
        for row in reader:
            if any(cell.strip() for cell in row):
                row_count += 1
        return row_count


def collect_pair_specs(pairs_dir: Path) -> list[PairSpec]:
    """Return all pair CSVs found in the user-provided directory."""
    if not pairs_dir.exists():
        raise click.ClickException(f'Pairs directory does not exist: {pairs_dir}')
    if not pairs_dir.is_dir():
        raise click.ClickException(f'Pairs path is not a directory: {pairs_dir}')

    csv_paths = []
    for root, _, files in os.walk(pairs_dir):
        for fname in files:
            if not fname.endswith('.csv'):
                continue
            if fname.endswith('_sim.csv'):
                continue
            csv_paths.append(Path(root) / fname)

    csv_paths.sort(key=lambda path: os.path.relpath(path, pairs_dir))
    if not csv_paths:
        raise click.ClickException(
            f'No CSV files found in pairs directory: {pairs_dir}')

    pair_specs = []
    for csv_path in csv_paths:
        rel_path = Path(os.path.relpath(csv_path, pairs_dir))
        output_name = f'{rel_path.stem}_sim{rel_path.suffix}'
        output_rel_path = rel_path.with_name(output_name)
        pair_specs.append(PairSpec(
            csv_path=str(csv_path),
            output_rel_path=str(output_rel_path),
            row_count=count_csv_rows(csv_path)))

    return pair_specs


def validate_columns(fieldnames: Optional[Iterable[str]], required: set[str],
                     csv_path: Path) -> None:
    """Validate that a CSV contains the required columns."""
    if fieldnames is None:
        raise ValueError(f'CSV file is empty: {csv_path}')

    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(
            f'Missing required columns in {csv_path}: {", ".join(missing)}')


def normalize_key(path_value: str, address_value: str) -> Tuple[str, str]:
    """Normalize the merge key used by the notebook."""
    return path_value.strip(), address_value.strip().lower()


def is_repeated_fss_header(row: dict[str, str]) -> bool:
    """Return True when the current row is an accidental repeated header."""
    return (
        row.get('path', '').strip() == 'path' or
        row.get('branching_nodes', '').strip() == 'branching_nodes')


def load_fss_index(fss_csv: str) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Load one raw FunctionSimSearch CSV into a dictionary."""
    csv_path = Path(fss_csv)
    cache_key = str(csv_path.resolve())
    cached = _FSS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fss_index: Dict[Tuple[str, str], Tuple[int, int]] = {}
    with csv_path.open(newline='') as f_in:
        reader = csv.DictReader(f_in)
        validate_columns(reader.fieldnames, FSS_REQUIRED_COLUMNS, csv_path)

        for row in reader:
            if not any((value or '').strip() for value in row.values()):
                continue
            if is_repeated_fss_header(row):
                continue

            key = normalize_key(row['path'], row['address'])
            hashes = (int(row['hashes0']), int(row['hashes1']))

            existing = fss_index.get(key)
            if existing is not None and existing != hashes:
                raise ValueError(
                    'Conflicting duplicate FunctionSimSearch entry in '
                    f'{csv_path}: {row["path"]} @ {row["address"]}')
            fss_index[key] = hashes

    _FSS_CACHE[cache_key] = fss_index
    return fss_index


def hamming_similarity(hash_pair_1: Tuple[int, int],
                       hash_pair_2: Tuple[int, int]) -> float:
    """Return the normalized 128-bit Hamming similarity."""
    diff = (
        (hash_pair_1[0] ^ hash_pair_2[0]).bit_count() +
        (hash_pair_1[1] ^ hash_pair_2[1]).bit_count())
    return 1.0 - (diff / 128.0)


def format_similarity(score: float) -> str:
    """Format the similarity score in a stable CSV-friendly form."""
    return f'{score:.17g}'


def format_missing_example(row: dict[str, str], which: str) -> str:
    """Return a short example string for a missing hash entry."""
    path_value = row[f'idb_path_{which}'].strip()
    fva_value = row[f'fva_{which}'].strip()
    return f'{path_value} @ {fva_value}'


def convert_one_task(
        task: ConversionTask,
        allow_missing: bool = False,
        progress_callback: Optional[Callable[[int], None]] = None,
        progress_step: int = 2048) -> dict[str, object]:
    """Convert one pairs CSV against one raw FunctionSimSearch CSV."""
    pair_path = Path(task.pair_csv)
    output_path = Path(task.output_csv)
    temp_path = output_path.with_suffix(output_path.suffix + '.tmp')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fss_index = load_fss_index(task.fss_csv)

    processed_rows = 0
    missing_rows = 0
    missing_examples: list[str] = []
    buffered_progress = 0

    try:
        with pair_path.open(newline='') as f_in, temp_path.open(
                'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            validate_columns(reader.fieldnames, PAIR_REQUIRED_COLUMNS, pair_path)

            fieldnames = list(reader.fieldnames or [])
            if 'sim' not in fieldnames:
                fieldnames.append('sim')
            writer = csv.DictWriter(
                f_out,
                fieldnames=fieldnames,
                lineterminator='\n')
            writer.writeheader()

            for row in reader:
                if not any((value or '').strip() for value in row.values()):
                    continue

                left_key = normalize_key(row['idb_path_1'], row['fva_1'])
                right_key = normalize_key(row['idb_path_2'], row['fva_2'])
                left_hashes = fss_index.get(left_key)
                right_hashes = fss_index.get(right_key)

                if left_hashes is None or right_hashes is None:
                    missing_rows += 1
                    if len(missing_examples) < 5:
                        if left_hashes is None:
                            missing_examples.append(
                                format_missing_example(row, '1'))
                        if right_hashes is None and len(missing_examples) < 5:
                            missing_examples.append(
                                format_missing_example(row, '2'))
                    similarity = ''
                else:
                    similarity = format_similarity(
                        hamming_similarity(left_hashes, right_hashes))

                if allow_missing or similarity:
                    row['sim'] = similarity
                    writer.writerow(row)

                processed_rows += 1
                buffered_progress += 1
                if progress_callback is not None and buffered_progress >= progress_step:
                    progress_callback(buffered_progress)
                    buffered_progress = 0

        if progress_callback is not None and buffered_progress:
            progress_callback(buffered_progress)

        if missing_rows and not allow_missing:
            example_text = ', '.join(missing_examples)
            raise ValueError(
                f'Missing {missing_rows} FunctionSimSearch hash lookups while '
                f'processing {pair_path.name} with {Path(task.fss_csv).name}. '
                f'Examples: {example_text}')

        temp_path.replace(output_path)
        return {
            'fss_csv': task.fss_csv,
            'pair_csv': task.pair_csv,
            'output_csv': str(output_path),
            'rows': processed_rows,
            'missing_rows': missing_rows,
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()


def worker_convert_task(task: ConversionTask, allow_missing: bool) -> dict[str, object]:
    """Process-pool wrapper for convert_one_task."""
    return convert_one_task(task, allow_missing=allow_missing)


def build_tasks(fss_csvs: Iterable[Path], pair_specs: Iterable[PairSpec],
                output_dir: Path) -> list[ConversionTask]:
    """Build one conversion task per pairs CSV."""
    tasks = []
    for fss_csv in sorted(fss_csvs):
        for pair_spec in pair_specs:
            output_csv = output_dir / pair_spec.output_rel_path
            tasks.append(ConversionTask(
                fss_csv=str(fss_csv),
                pair_csv=pair_spec.csv_path,
                output_csv=str(output_csv),
                row_count=pair_spec.row_count))
    return tasks


def run_tasks(tasks: list[ConversionTask], jobs: int, allow_missing: bool,
              verbose: bool) -> list[dict[str, object]]:
    """Run conversion tasks with a progress bar and optional multiprocessing."""
    results = []
    total_rows = sum(task.row_count for task in tasks)
    use_task_units = total_rows == 0
    progress_length = len(tasks) if use_task_units else total_rows

    with click.progressbar(length=progress_length,
                           label='[D] Converting FunctionSimSearch results') as progress_bar:
        if jobs == 1:
            for task in tasks:
                if use_task_units:
                    result = convert_one_task(task, allow_missing=allow_missing)
                    progress_bar.update(1)
                else:
                    result = convert_one_task(
                        task,
                        allow_missing=allow_missing,
                        progress_callback=progress_bar.update)
                results.append(result)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(worker_convert_task, task, allow_missing): task
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    progress_bar.update(1 if use_task_units else result['rows'])
                    results.append(result)

    if verbose:
        for result in sorted(results, key=lambda x: x['output_csv']):
            click.echo(
                f"[D] {result['pair_csv']} + {result['fss_csv']} => "
                f"{result['output_csv']} "
                f"(rows={result['rows']}, missing={result['missing_rows']})")

    return results


@click.command()
@click.option(
    '-i',
    '--fss-input',
    required=True,
    help='Raw FunctionSimSearch CSV file, or a directory containing '
         f'{TARGET_FSS_CSV_NAME}.')
@click.option(
    '-o',
    '--output-dir',
    required=True,
    help='Directory where the converted similarity CSVs will be written.')
@click.option(
    '-p',
    '--pairs-dir',
    required=True,
    help='Directory containing the pair CSV files to convert.')
@click.option(
    '-n',
    '--jobs',
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help='Number of worker processes. Use 0 for auto.')
@click.option(
    '--allow-missing',
    is_flag=True,
    help='Write blank sim values instead of failing on missing hashes.')
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Print per-output completion details.')
def main(fss_input: str, output_dir: str, pairs_dir: str, jobs: int,
         allow_missing: bool, verbose: bool) -> None:
    """Convert raw FunctionSimSearch hashes into pairwise similarity CSVs."""
    output_dir_path = Path(output_dir)
    fss_input_path = Path(fss_input)
    pairs_dir_path = Path(pairs_dir)

    pair_specs = collect_pair_specs(pairs_dir_path)
    fss_csvs = collect_fss_csvs(fss_input_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(fss_csvs, pair_specs, output_dir_path)
    effective_jobs = get_parallel_jobs(jobs, len(tasks))

    click.echo(f'[D] Raw FSS input: {fss_input_path}')
    click.echo(f'[D] Pairs dir: {pairs_dir_path}')
    click.echo(f'[D] Output dir: {output_dir_path}')
    click.echo(f'[D] Pairs CSVs: {len(pair_specs)}')
    click.echo(f'[D] Raw FSS CSVs: {len(fss_csvs)}')
    click.echo(f'[D] Worker processes: {effective_jobs}')

    results = run_tasks(
        tasks,
        jobs=effective_jobs,
        allow_missing=allow_missing,
        verbose=verbose)

    click.echo(
        f'[D] Wrote {len(results)} converted similarity CSV file(s) to '
        f'{output_dir_path}')


if __name__ == '__main__':
    main()
