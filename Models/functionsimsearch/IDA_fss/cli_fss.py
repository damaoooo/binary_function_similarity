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
#  cli_fss.py - Call IDA_fss.py IDA script.                                  #
#                                                                            #
##############################################################################

import concurrent.futures
import json
import os
import re
import subprocess
import time

import click

from os import getenv
from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import isfile
from os.path import join
from os.path import normpath
from shutil import which

IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_fss.py')
REPO_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
IDA_EXECUTABLES = ("idat", "idat64")
AUTO_MAX_JOBS = 16


def expand_ida_candidates(path):
    """Yield possible IDA executable paths from a file or directory input."""
    if not path:
        return

    if isdir(path):
        for executable in IDA_EXECUTABLES:
            yield join(path, executable)
        return

    yield path


def resolve_ida_path(explicit_path=None):
    """Resolve the first valid IDA executable path."""
    seen = set()

    candidates = []
    candidates.extend(expand_ida_candidates(explicit_path))
    candidates.extend(expand_ida_candidates(getenv("IDA_PATH")))

    for env_name in ("IDADIR", "IDA_DIR"):
        candidates.extend(expand_ida_candidates(getenv(env_name)))

    for executable in IDA_EXECUTABLES:
        resolved = which(executable)
        if resolved:
            candidates.append(resolved)

    for candidate in candidates:
        candidate = abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if isfile(candidate):
            return candidate

    return None


def get_parallel_jobs(jobs, task_count):
    """Return the number of concurrent IDA processes to launch."""
    if task_count <= 0:
        return 1

    if jobs == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, cpu_count, AUTO_MAX_JOBS))

    return min(jobs, task_count)


def sanitize_log_name(idb_rel_path):
    """Create a filesystem-safe log filename from the relative IDB path."""
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', idb_rel_path).strip('._')
    return safe_name or "ida_fss"


def build_ida_command(ida_path, json_path, output_dir, use_capstone,
                      log_path, idb_rel_path, idb_path):
    """Build the headless IDA command line for a single IDB."""
    return [
        ida_path,
        '-A',
        '-L{}'.format(log_path),
        '-S{}'.format(IDA_PLUGIN),
        '-Ofss:{}:{}:{}:{}'.format(
            json_path,
            idb_rel_path,
            output_dir,
            str(use_capstone)),
        idb_path,
    ]


def run_single_idb(ida_path, json_path, output_dir, log_dir, use_capstone,
                   idb_rel_path):
    """Run the IDA extraction for a single IDB and return a result dict."""
    idb_path = normpath(join(REPO_PATH, idb_rel_path))
    log_path = join(log_dir, "{}.log".format(sanitize_log_name(idb_rel_path)))

    result = {
        'idb_rel_path': idb_rel_path,
        'idb_path': idb_path,
        'log_path': log_path,
        'cmd': None,
        'returncode': 1,
        'stdout': '',
        'stderr': '',
        'elapsed_time': 0.0,
        'error': None,
    }

    if not isfile(idb_path):
        result['error'] = "{} does not exist".format(idb_path)
        return result

    cmd = build_ida_command(
        ida_path=ida_path,
        json_path=json_path,
        output_dir=output_dir,
        use_capstone=use_capstone,
        log_path=log_path,
        idb_rel_path=idb_rel_path,
        idb_path=idb_path)
    result['cmd'] = cmd

    start_time = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    result['elapsed_time'] = time.time() - start_time
    result['returncode'] = proc.returncode
    result['stdout'] = proc.stdout
    result['stderr'] = proc.stderr
    return result


@click.command()
@click.option('-j', '--json-path', required=True,
              help='JSON file with selected functions.')
@click.option('-o', '--output-dir', required=True,
              help='Output directory.')
@click.option('-c', '--use-capstone', is_flag=True)
@click.option('--ida-path', default=None,
              help='Path to idat/idat64, or the IDA installation directory.')
@click.option('-n', '--jobs', default=0, show_default=True,
              type=click.IntRange(min=0),
              help='Number of parallel IDA processes. Use 0 for auto.')
@click.option('--log-dir', default=None,
              help='Directory for per-IDB IDA logs. Defaults to OUTPUT_DIR/logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Print per-IDB success details in addition to the progress bar.')
def main(json_path, output_dir, use_capstone, ida_path, jobs, log_dir, verbose):
    """Call IDA_fss.py IDA script."""
    try:
        if not isfile(json_path):
            print("[!] Error: {} does not exist".format(json_path))
            return

        ida_path = resolve_ida_path(ida_path)
        if not ida_path:
            print("[!] Error: unable to find a valid IDA executable")
            print("Use --ida-path /full/path/to/idat or export IDA_PATH/IDADIR")
            return

        json_path = abspath(json_path)
        output_dir = abspath(output_dir)
        log_dir = abspath(log_dir) if log_dir else join(output_dir, "logs")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        print("[D] JSON path: {}".format(json_path))
        print("[D] Output directory: {}".format(output_dir))
        print("[D] Log directory: {}".format(log_dir))
        print("[D] IDA path: {}".format(ida_path))
        print("[D] Use Capstone: {}".format(use_capstone))

        with open(json_path) as f_in:
            jj = json.load(f_in)

        task_count = len(jj)
        jobs = get_parallel_jobs(jobs, task_count)
        print("[D] Parallel jobs: {}".format(jobs))

        if not task_count:
            print("[!] Error: no IDBs found in {}".format(json_path))
            return

        success_cnt, error_cnt = 0, 0
        start_time = time.time()
        result_messages = []

        with click.progressbar(length=task_count, label='[D] Processing IDBs') as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        run_single_idb,
                        ida_path,
                        json_path,
                        output_dir,
                        log_dir,
                        use_capstone,
                        idb_rel_path): idb_rel_path
                    for idb_rel_path in jj.keys()
                }

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    progress_bar.update(1)

                    message_lines = [
                        "",
                        "[D] Processing: {}".format(result['idb_rel_path']),
                        "[D] IDB full path: {}".format(result['idb_path']),
                        "[D] Log path: {}".format(result['log_path']),
                    ]

                    if result['error']:
                        message_lines.append("[!] Error: {}".format(result['error']))
                        result_messages.append("\n".join(message_lines))
                        error_cnt += 1
                        continue

                    if verbose:
                        message_lines.append("[D] cmd: {}".format(result['cmd']))

                    if result['returncode'] == 0:
                        if verbose:
                            message_lines.append(
                                "[D] {}: success ({:.2f}s)".format(
                                    result['idb_path'], result['elapsed_time']))
                            result_messages.append("\n".join(message_lines))
                        success_cnt += 1
                        continue

                    message_lines.append("[!] Error in {} (returncode={})".format(
                        result['idb_path'], result['returncode']))
                    if result['stderr'].strip():
                        message_lines.append("[!] stderr: {}".format(
                            result['stderr'].strip()))
                    if result['stdout'].strip():
                        message_lines.append("[!] stdout: {}".format(
                            result['stdout'].strip()))
                    message_lines.append("[!] See IDA log: {}".format(result['log_path']))
                    result_messages.append("\n".join(message_lines))
                    error_cnt += 1

        end_time = time.time()
        for message in result_messages:
            print(message)
        print("[D] Elapsed time: {}".format(end_time - start_time))
        print("\n# IDBs correctly processed: {}".format(success_cnt))
        print("# IDBs error: {}".format(error_cnt))

    except Exception as e:
        print("[!] Exception in cli_fss\n{}".format(e))


if __name__ == '__main__':
    main()
