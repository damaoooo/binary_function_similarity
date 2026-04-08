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
#  IDA_catalog1.py - Catalog1 IDA plugin implementation.                     #
#                                                                            #
##############################################################################

import ida_bytes
import ida_funcs
import ida_gdl
import ida_loader
import ida_pro
import json
import os
import time

from catalog1.catalog_fast import sign
from contextlib import ExitStack
from collections import namedtuple

COLUMNS = ['path', 'address', 'size', 'catalog_hash_list', 'time']
BasicBlock = namedtuple('BasicBlock', ['va', 'size'])


def get_basic_blocks(fva):
    """Return the list of BasicBlock for a given function."""
    bb_list = []
    func = ida_funcs.get_func(fva)
    if func is None:
        return bb_list
    for bb in ida_gdl.FlowChart(func):
        # NOTE: a BB may have size 0.
        bb_list.append(
            BasicBlock(
                va=bb.start_ea,
                size=bb.end_ea - bb.start_ea))
    return bb_list


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
        raise ValueError("No signature sizes provided")

    return parsed_sizes


def get_output_csv_path(output_csv_base, sig_size):
    """Return the output CSV path for a given signature size."""
    base_name, extension = os.path.splitext(output_csv_base)
    if not extension:
        extension = ".csv"
    return "{}_{}{}".format(base_name, sig_size, extension)


def get_function_binary_data(fva):
    """Read and concatenate the bytes of all basic blocks in a function."""
    func_binary_data = b""
    for bb in sorted(get_basic_blocks(fva)):
        bb_data = ida_bytes.get_bytes(bb.va, bb.size)
        if bb_data:
            func_binary_data += bb_data
    return func_binary_data


def run_catalog1(idb_path, fva_list, sig_sizes, output_csv_base):
    """Compute the Catalog1 hash for each selected function and size."""
    if isinstance(sig_sizes, str):
        sig_sizes = parse_sig_sizes(sig_sizes)

    with ExitStack() as stack:
        csv_files = {}
        for sig_size in sig_sizes:
            output_csv = get_output_csv_path(output_csv_base, sig_size)
            write_header = not os.path.isfile(output_csv)
            csv_out = stack.enter_context(
                open(output_csv, "a" if not write_header else "w"))
            if write_header:
                csv_out.write(",".join(COLUMNS) + "\n")
            csv_files[sig_size] = csv_out
            print("[D] Output CSV ({}): {}".format(sig_size, output_csv))

        # For each function in the list
        for fva in fva_list:
            try:
                func_binary_data = get_function_binary_data(fva)
                function_size = len(func_binary_data)

                for sig_size in sig_sizes:
                    start_time = time.time()

                    if function_size < 4:
                        catalog1_signature = "min_function_size_error"
                    else:
                        catalog1_list = sign(func_binary_data, sig_size)
                        catalog1_signature = ';'.join(
                            [str(x) for x in catalog1_list])

                    elapsed_time = time.time() - start_time
                    data = [idb_path,
                            hex(fva).strip("L"),
                            function_size,
                            catalog1_signature,
                            elapsed_time]

                    csv_files[sig_size].write(
                        ",".join([str(x) for x in data]) + "\n")

            except Exception as e:
                print("[!] Exception: skipping function fva: %d" % fva)
                print(e)


if __name__ == '__main__':
    plugin_options = ida_loader.get_plugin_options("catalog1")
    if not plugin_options:
        print("[!] -Ocatalog1 option is missing")
        ida_pro.qexit(1)

    plugin_options = plugin_options.split(':')
    if len(plugin_options) != 4:
        print("[!] -Ocatalog1:INPUT_JSON:IDB_PATH:SIG_SIZES:OUTPUT_CSV_BASE is required")
        ida_pro.qexit(1)

    input_json = plugin_options[0]
    idb_path = plugin_options[1]
    sig_sizes = parse_sig_sizes(plugin_options[2])
    output_csv_base = plugin_options[3]

    with open(input_json) as f_in:
        selected_functions = json.load(f_in)

    if idb_path not in selected_functions:
        print("[!] Error! IDB path (%s) not in %s" % (idb_path, input_json))
        ida_pro.qexit(1)

    fva_list = selected_functions[idb_path]
    print("[D] Found %d addresses" % len(fva_list))
    print("[D] Signature sizes: {}".format(sig_sizes))

    run_catalog1(idb_path, fva_list, sig_sizes, output_csv_base)
    ida_pro.qexit(0)
