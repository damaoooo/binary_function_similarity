#!/usr/bin/env python3

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
#  fss_simhasher.py - Compute the simhash for all the  functions in input.   #
#                                                                            #
##############################################################################

import concurrent.futures
import click
import importlib
import json
import os
import tempfile
import time
import traceback

from os.path import isdir
from os.path import isfile
from os.path import join
from os.path import basename

AUTO_MAX_JOBS = 16
FUNCTIONSIMSEARCH_MODULE = None

CSV_COLUMNS = [
    'path',
    'address',
    'num_nodes',
    'branching_nodes',
    'hashes0',
    'hashes1',
    'time']


def get_functionsimsearch():
    """Load the functionsimsearch module on demand."""
    global FUNCTIONSIMSEARCH_MODULE
    if FUNCTIONSIMSEARCH_MODULE is None:
        FUNCTIONSIMSEARCH_MODULE = importlib.import_module("functionsimsearch")
    return FUNCTIONSIMSEARCH_MODULE


def construct_flowgraph(nodes, edges, instructions_dict):
    """Construct a functionsimsearch flowgraph."""
    functionsimsearch = get_functionsimsearch()
    flowgraph = functionsimsearch.FlowgraphWithInstructions()

    # Add the nodes
    for node_ea in nodes:
        flowgraph.add_node(int(node_ea))

    # Add the instructions
    for node_ea, ins in instructions_dict.items():
        ins_t = tuple([(v[0], tuple(v[1])) for v in ins])
        flowgraph.add_instructions(int(node_ea), ins_t)

    # Add the edges
    for edge in edges:
        flowgraph.add_edge(edge[0], edge[1])

    return flowgraph


def create_simhasher(imm_w, mnem_w, graph_w):
    """Create a SimHasher with the requested weights."""
    functionsimsearch = get_functionsimsearch()

    # Initialize the simhasher with a given weight configuration
    return functionsimsearch.SimHasher(
        immediate_weight=imm_w,
        mnem_weight=mnem_w,
        graphlet_weight=graph_w,
    )


def list_input_jsons(input_dir):
    """Return sorted input JSON paths to process."""
    return sorted([
        join(input_dir, json_name)
        for json_name in os.listdir(input_dir)
        if json_name.endswith('_fss.json') and isfile(join(input_dir, json_name))
    ])


def get_parallel_jobs(jobs, task_count):
    """Return the effective number of worker processes."""
    if task_count <= 0:
        return 1

    if jobs == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, cpu_count, AUTO_MAX_JOBS))

    return min(jobs, task_count)


def compute_simhashes_for_json(json_path, output_dir, imm_w, mnem_w, graph_w):
    """Compute simhashes for one input JSON and write one CSV."""
    sim_hasher = create_simhasher(imm_w, mnem_w, graph_w)

    json_name = basename(json_path)
    csv_name = json_name.replace(
        '_fss.json',
        f'_IMM:{imm_w:.2f}_MNEM:{mnem_w:.2f}_GRAPH:{graph_w:.2f}.csv')
    csv_path = join(output_dir, csv_name)

    function_count = 0
    failure_count = 0

    with open(json_path) as f_in:
        j_in = json.load(f_in)

    with open(csv_path, "w") as f_out:
        f_out.write(",".join(CSV_COLUMNS) + "\n")

        # Iterate over different "IDBs". Usually, 1 JSON -> 1 IDB.
        for idb_path in j_in.keys():
            # Iterate over each function
            for fva in j_in[idb_path].keys():
                function_count += 1
                try:
                    j_data = j_in[idb_path][fva]
                    nodes = j_data.get('nodes')
                    edges = j_data.get('edges')
                    instructions_dict = j_data.get('instructions')

                    start_time = time.perf_counter()

                    # Get the flowgraph for the current function
                    flowgraph = construct_flowgraph(
                        nodes,
                        edges,
                        instructions_dict)

                    flowgraph_size = flowgraph.size()
                    branching_nodes = flowgraph.number_of_branching_nodes()
                    hashes = sim_hasher.calculate_hash(flowgraph)
                    elapsed_time = time.perf_counter() - start_time

                    # Save the simhash to a CSV file
                    columns = [idb_path,
                               fva,
                               flowgraph_size,
                               branching_nodes,
                               hashes[0],
                               hashes[1],
                               elapsed_time]
                    f_out.write(",".join([str(x) for x in columns]) + "\n")

                except Exception:
                    failure_count += 1
                    print("[!] Exception: skipping function: {}".format(fva))
                    print('tb: {}'.format(traceback.format_exc()))

    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'functions': function_count,
        'failures': failure_count,
    }


def aggregate_csvs(temp_output_dir, output_csv_path):
    """Merge per-JSON CSVs into one configuration-wide CSV."""
    output_csv_lines = []

    for temp_csv_name in sorted(os.listdir(temp_output_dir)):
        with open(join(temp_output_dir, temp_csv_name)) as f:
            lines = list(filter(lambda x: x.strip(), f.read().split('\n')))
            if not lines:
                continue
            if not len(output_csv_lines):
                output_csv_lines.append(lines[0])
            output_csv_lines.extend(lines[1:])

    if not output_csv_lines:
        output_csv_lines.append(",".join(CSV_COLUMNS))

    with open(output_csv_path, 'w') as f:
        f.write('\n'.join(output_csv_lines))


def compute_simhashes(input_dir, output_dir, imm_w, mnem_w, graph_w, jobs,
                      verbose):
    """Compute the simhashes for all input JSONs."""
    json_paths = list_input_jsons(input_dir)
    task_count = len(json_paths)

    if not task_count:
        return []

    jobs = get_parallel_jobs(jobs, task_count)
    print(
        f'[D] Config IMM={imm_w:.2f} MNEM={mnem_w:.2f} GRAPH={graph_w:.2f} '
        f'with {jobs} worker(s)')

    results = []
    label = (
        f'[D] Hashing IMM={imm_w:.2f} MNEM={mnem_w:.2f} GRAPH={graph_w:.2f}')
    with click.progressbar(length=task_count, label=label) as progress_bar:
        if jobs == 1:
            for json_path in json_paths:
                result = compute_simhashes_for_json(
                    json_path,
                    output_dir,
                    imm_w,
                    mnem_w,
                    graph_w)
                progress_bar.update(1)
                results.append(result)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        compute_simhashes_for_json,
                        json_path,
                        output_dir,
                        imm_w,
                        mnem_w,
                        graph_w): json_path
                    for json_path in json_paths
                }

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    progress_bar.update(1)
                    results.append(result)

    if verbose:
        for result in sorted(results, key=lambda x: x['json_path']):
            print(
                f"[D] {result['json_path']} => {result['csv_path']} "
                f"(functions={result['functions']}, failures={result['failures']})")

    return results


@click.command()
@click.option('-i', 'input_dir', default='/input')
@click.option('-o', 'output_dir', default='/output')
@click.option('-n', '--jobs', default=0, show_default=True,
              type=click.IntRange(min=0),
              help='Number of worker processes. Use 0 for auto.')
@click.option('-v', '--verbose', is_flag=True,
              help='Print per-input JSON completion details.')
def main(input_dir, output_dir, jobs, verbose):
    """Compute the simhash for different weight configs."""
    if not isdir(input_dir):
        print("[!] Error: {} does not exist".format(input_dir))
        return

    if not isdir(output_dir):
        print("[!] Error: {} does not exist".format(output_dir))
        return

    print(f'[D] Input dir: {input_dir}')
    print(f'[D] Output dir: {output_dir}')
    print(f'[D] Requested jobs: {jobs}')

    configurations = [
        # immediate, mnemonic, graphlet
        (4, 0.05, 1),
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]

    # Iterate over the weight configurations
    for imm_w, mnem_w, graph_w in configurations:
        with tempfile.TemporaryDirectory() as t_output_dir:
            results = compute_simhashes(
                input_dir,
                t_output_dir,
                imm_w,
                mnem_w,
                graph_w,
                jobs,
                verbose)
            output_csv_path = join(
                output_dir,
                f'IMM:{imm_w:.2f}_MNEM:{mnem_w:.2f}_GRAPH:{graph_w:.2f}.csv')
            aggregate_csvs(t_output_dir, output_csv_path)

            total_functions = sum(result['functions'] for result in results)
            total_failures = sum(result['failures'] for result in results)
            print(
                f'[D] Wrote {output_csv_path} '
                f'(functions={total_functions}, failures={total_failures})')


if __name__ == '__main__':
    main()
