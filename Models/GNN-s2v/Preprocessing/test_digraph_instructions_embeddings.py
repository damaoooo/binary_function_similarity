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
#  test_digraph_instructions_embeddings.py                                   #
#                                                                            #
##############################################################################

import json
import os
import shutil
import unittest

from digraph_instructions_embeddings import main
from click.testing import CliRunner


class TestDigraphInstructionsEmbeddings(unittest.TestCase):
    def test_digraph_instructions_embeddings(self):
        input_dir = "/input/acfg_disasm"
        output_dir = "/output"
        embeddings_file = "/instruction_embeddings/ins2id.json"
        gt_path = "/input/digraph_instructions_embeddings_200_gt.json"

        runner = CliRunner()

        result = runner.invoke(
            main, ["-i", input_dir, "-d", embeddings_file, "-o", output_dir]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.isdir(output_dir))

        with open(gt_path) as f_in:
            j_gt = json.load(f_in)

        with open(
            os.path.join(output_dir, "digraph_instructions_embeddings_200.json")
        ) as f_in:
            j_o = json.load(f_in)
        self.assertDictEqual(j_gt, j_o)

        # Cleanup files
        for file in os.scandir(output_dir):
            if os.path.isfile(file.path):
                os.remove(file.path)
            else:
                shutil.rmtree(file.path)
