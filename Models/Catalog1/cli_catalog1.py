#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
#  cli_catalog1.py - Call IDA_catalog1 IDA script.                           #
#                                                                            #
##############################################################################

import click
import json
import subprocess
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from os import cpu_count
from os import getenv
from os.path import abspath
from os.path import basename
from os.path import dirname
from os.path import isfile
from os.path import join
from os.path import splitext

IDA_PATH = getenv("IDA_PATH", "/home/damaoooo/ida-pro-9.3/idat")
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_catalog1.py')
REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
LOG_PATH = "catalog1_log.txt"
CSV_HEADER = "path,address,size,catalog_hash_list,time\n"
DEFAULT_SIG_SIZES = [16, 32, 64, 128]


def parse_sig_sizes(sig_sizes):
    """Parse a comma-separated list of signature sizes."""
    parsed_sizes = []
    for item in sig_sizes.split(','):
        item = item.strip()
        if not item:
            continue
        sig_size = int(item)
        if sig_size not in parsed_sizes:
            parsed_sizes.append(sig_size)

    if not parsed_sizes:
        raise click.BadParameter("At least one signature size is required")

    return parsed_sizes


def get_output_csv_path(output_csv, sig_size):
    """Return the CSV path for a given signature size."""
    base_name, extension = splitext(output_csv)
    if not extension:
        extension = ".csv"
    return "{}_{}{}".format(base_name, sig_size, extension)


def build_task(index, json_path, idb_rel_path, sig_sizes, temp_dir):
    """Build the execution metadata for one IDB."""
    task_name = "task_{:05d}_{}".format(
        index, basename(idb_rel_path).replace('.', '_'))
    return {
        "idb_rel_path": idb_rel_path,
        "idb_path": join(REPO_PATH, idb_rel_path),
        "json_path": json_path,
        "sig_sizes": sig_sizes,
        "sig_sizes_arg": ",".join([str(x) for x in sig_sizes]),
        "output_base": join(temp_dir, task_name),
        "log_path": join(temp_dir, "{}.log".format(task_name)),
    }


def build_error_result(task, error_message):
    """Build a failed task result."""
    return {
        "idb_rel_path": task["idb_rel_path"],
        "idb_path": task["idb_path"],
        "returncode": 1,
        "elapsed_time": 0.0,
        "stdout": "",
        "stderr": error_message,
        "log_path": task["log_path"],
        "output_paths": {},
    }


def run_ida_task(task):
    """Run one IDA batch job for one IDB."""
    if not isfile(task["idb_path"]):
        return build_error_result(
            task, "[!] Error: {} does not exist".format(task["idb_path"]))

    cmd = [IDA_PATH,
           '-A',
           '-L{}'.format(task["log_path"]),
           '-S{}'.format(IDA_PLUGIN),
           '-Ocatalog1:{}:{}:{}:{}'.format(
               task["json_path"],
               task["idb_rel_path"],
               task["sig_sizes_arg"],
               task["output_base"]),
           task["idb_path"]]

    try:
        start_time = time.time()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        elapsed_time = time.time() - start_time
    except Exception as e:
        return build_error_result(task, str(e))

    output_paths = {
        sig_size: get_output_csv_path(task["output_base"], sig_size)
        for sig_size in task["sig_sizes"]
    }

    return {
        "idb_rel_path": task["idb_rel_path"],
        "idb_path": task["idb_path"],
        "returncode": proc.returncode,
        "elapsed_time": elapsed_time,
        "stdout": stdout,
        "stderr": stderr,
        "log_path": task["log_path"],
        "output_paths": output_paths,
    }


def merge_output_csvs(output_csv, sig_sizes, idb_rel_paths, task_results):
    """Merge task-local CSVs into the final per-signature CSV files."""
    for sig_size in sig_sizes:
        final_output = get_output_csv_path(output_csv, sig_size)
        with open(final_output, "w") as f_out:
            f_out.write(CSV_HEADER)
            for idb_rel_path in idb_rel_paths:
                result = task_results.get(idb_rel_path)
                if result is None or result["returncode"] != 0:
                    continue

                temp_output = result["output_paths"].get(sig_size)
                if not temp_output or not isfile(temp_output):
                    continue

                with open(temp_output) as f_in:
                    lines = f_in.readlines()
                if len(lines) > 1:
                    f_out.writelines(lines[1:])


def write_summary_log(log_path, jobs, sig_sizes, total_elapsed, idb_rel_paths,
                      task_results):
    """Write one merged log file for all IDA tasks."""
    with open(log_path, "w") as f_out:
        f_out.write("workers: {}\n".format(jobs))
        f_out.write("sig_sizes: {}\n".format(",".join([str(x) for x in sig_sizes])))
        f_out.write("elapsed_time: {}\n".format(total_elapsed))

        for idb_rel_path in idb_rel_paths:
            result = task_results.get(idb_rel_path)
            if result is None:
                continue

            f_out.write("\n===== {} =====\n".format(idb_rel_path))
            f_out.write("returncode: {}\n".format(result["returncode"]))
            f_out.write("elapsed_time: {}\n".format(result["elapsed_time"]))

            if isfile(result["log_path"]):
                with open(result["log_path"]) as f_in:
                    log_content = f_in.read().strip()
                if log_content:
                    f_out.write("[ida-log]\n{}\n".format(log_content))

            if result["stdout"].strip():
                f_out.write("[stdout]\n{}\n".format(result["stdout"].strip()))

            if result["stderr"].strip():
                f_out.write("[stderr]\n{}\n".format(result["stderr"].strip()))


@click.command()
@click.option('-j', '--json-path', required=True,
              help='JSON file with selected functions.')
@click.option("-o", "--output-csv", required=True,
              help="Path to the output CSV file")
@click.option("-n", "--jobs", default=cpu_count() or 1, show_default=True,
              type=click.IntRange(1, None),
              help="Number of parallel IDA workers.")
@click.option("--sig-sizes",
              default=",".join([str(x) for x in DEFAULT_SIG_SIZES]),
              show_default=True,
              help="Comma-separated Catalog1 signature sizes.")
def main(json_path, output_csv, jobs, sig_sizes):
    """Call IDA_catalog1 IDA script."""
    try:
        if not isfile(IDA_PATH):
            print("[!] Error: IDA_PATH:{} not valid".format(IDA_PATH))
            print("Use 'export IDA_PATH=/full/path/to/idat'")
            return

        print("[D] JSON path: {}".format(json_path))
        print("[D] Output CSV: {}".format(output_csv))

        if not isfile(json_path):
            print("[!] Error: {} does not exist".format(json_path))
            return

        sig_sizes = parse_sig_sizes(sig_sizes)

        with open(json_path) as f_in:
            selected_functions = json.load(f_in)

        idb_rel_paths = list(selected_functions.keys())
        if not idb_rel_paths:
            print("[!] Error: no IDBs found in {}".format(json_path))
            return

        effective_jobs = min(jobs, len(idb_rel_paths))
        print("[D] Signature sizes: {}".format(sig_sizes))
        print("[D] Parallel workers: {}".format(effective_jobs))

        total_start_time = time.time()
        task_results = {}

        with tempfile.TemporaryDirectory(prefix="catalog1_") as temp_dir:
            tasks = [
                build_task(i, json_path, idb_rel_path, sig_sizes, temp_dir)
                for i, idb_rel_path in enumerate(idb_rel_paths)
            ]

            with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
                future_to_task = {
                    executor.submit(run_ida_task, task): task
                    for task in tasks
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = build_error_result(task, str(e))

                    task_results[result["idb_rel_path"]] = result

                    if result["returncode"] == 0:
                        print("[D] {}: success ({:.2f}s)".format(
                            result["idb_path"], result["elapsed_time"]))
                    else:
                        print("[!] Error in {} (returncode={})".format(
                            result["idb_path"], result["returncode"]))
                        if result["stderr"].strip():
                            print(result["stderr"].strip())
                        print("[!] Task log: {}".format(result["log_path"]))

            merge_output_csvs(output_csv, sig_sizes, idb_rel_paths, task_results)

            total_elapsed = time.time() - total_start_time
            write_summary_log(
                LOG_PATH,
                effective_jobs,
                sig_sizes,
                total_elapsed,
                idb_rel_paths,
                task_results)

        success_cnt = sum(
            1 for result in task_results.values() if result["returncode"] == 0)
        error_cnt = len(idb_rel_paths) - success_cnt

        print("[D] Elapsed time: {}".format(total_elapsed))
        for sig_size in sig_sizes:
            print("[D] Output CSV ({}): {}".format(
                sig_size, get_output_csv_path(output_csv, sig_size)))

        print("\n# IDBs correctly processed: {}".format(success_cnt))
        print("# IDBs error: {}".format(error_cnt))

    except Exception as e:
        print("[!] Exception in cli_catalog1\n{}".format(e))


if __name__ == '__main__':
    main()
