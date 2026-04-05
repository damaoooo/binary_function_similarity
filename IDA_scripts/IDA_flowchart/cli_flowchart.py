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
#  cli_flowchart.py - Call IDA_flowchart.py IDA script.                      #
#                                                                            #
##############################################################################

import click
import shutil
import subprocess
import tempfile
import multiprocessing
from tqdm import tqdm


from os import getcwd
from os import getenv
from os import walk
from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import isfile
from os.path import join
from os.path import relpath

IDA_PATH = getenv("IDA_PATH", "/home/damaoooo/ida-pro-9.1/idat")
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_flowchart.py')
REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
LOG_PATH = "flowchart_log.txt"
CSV_HEADER = (
    "idb_path,fva,func_name,start_ea,end_ea,bb_num,bb_list,hashopcodes"
)


def execute_command(cmd):
    """Execute a command."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    return proc.returncode == 0


def merge_partial_csvs(partial_csvs, output_csv):
    """Merge temporary partial CSV files into a single output file."""
    header_written = False
    with open(output_csv, "w") as out_fp:
        for partial_csv in partial_csvs:
            if not isfile(partial_csv):
                continue

            with open(partial_csv, "r") as in_fp:
                for line_idx, line in enumerate(in_fp):
                    if line_idx == 0:
                        if header_written:
                            continue
                        header_written = True
                    out_fp.write(line)

        if not header_written:
            # Keep CSV output valid even when there are no input IDBs.
            out_fp.write(CSV_HEADER + "\n")


@click.command()
@click.option("-i", "--idbs-folder", required=True,
              help="Path to the IDBs folder")
@click.option("-o", "--output-csv", required=True,
              help="Path to the output CSV file")
def main(idbs_folder, output_csv):
    """Call IDA_flowchart.py IDA script."""
    temp_output_dir = None
    try:
        if not isfile(IDA_PATH):
            print("[!] Error: IDA_PATH:{} not valid".format(IDA_PATH))
            print("Use 'export IDA_PATH=/full/path/to/idat64'")
            return

        print("[D] IDBs folder: {}".format(idbs_folder))
        print("[D] Output CSV: {}".format(output_csv))

        temp_output_dir = tempfile.mkdtemp(
            prefix="flowchart_parts_",
            dir=dirname(abspath(output_csv)))
        print("[D] Temporary CSV folder: {}".format(temp_output_dir))

        commands = []
        partial_csvs = []

        success_cnt, error_cnt = 0, 0
        for root, _, files in walk(idbs_folder):
            for f_name in files:
                if (not f_name.endswith(".i64")) and \
                        (not f_name.endswith(".idb")):
                    continue

                idb_path = join(root, f_name)
                print("\n[D] Processing: {}".format(idb_path))

                if not isfile(idb_path):
                    print("[!] Error: {} not exists".format(idb_path))
                    continue

                # Compute the normalized relative path from the main directory
                rel_idb_path = relpath(
                    join(getcwd(), root, f_name),  # absolute path if IDB
                    REPO_PATH)  # absolute path of the repo folder

                partial_csv = join(
                    temp_output_dir,
                    "flowchart_part_{:08d}.csv".format(len(commands)))

                cmd = [IDA_PATH,
                       '-A',
                       '-L{}'.format(LOG_PATH),
                       '-S{}'.format(IDA_PLUGIN),
                       '-Oflowchart:{}:{}'.format(
                           rel_idb_path,
                           partial_csv),
                       idb_path]

                # print("[D] cmd: {}".format(' '.join(cmd)))
                commands.append(cmd)
                partial_csvs.append(partial_csv)

        merged_partials = []
        if commands:
            bar = tqdm(total=len(commands), desc="Processing IDBs")
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

            rets = []
            for cmd in commands:
                p = pool.apply_async(
                    execute_command,
                    args=(cmd,),
                    callback=lambda x: bar.update(1))
                rets.append(p)

            pool.close()
            pool.join()

            for partial_csv, p in zip(partial_csvs, rets):
                if p.get():
                    success_cnt += 1
                    merged_partials.append(partial_csv)
                else:
                    error_cnt += 1
        else:
            print("[!] Warning: no IDBs found in {}".format(idbs_folder))

        merge_partial_csvs(merged_partials, output_csv)

        print("\n# IDBs correctly processed: {}".format(success_cnt))
        print("# IDBs error: {}".format(error_cnt))
        print("[D] Merged CSV written to: {}".format(output_csv))

    except Exception as e:
        print("[!] Exception in cli_flowchart\n{}".format(e))
    finally:
        if temp_output_dir and isdir(temp_output_dir):
            shutil.rmtree(temp_output_dir)


if __name__ == '__main__':
    main()
