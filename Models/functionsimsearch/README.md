# FSS: Function Sim Search
The FSS experiment is made via two steps. [The first](#part-1) is an IDA Pro plugin that takes as input a JSON specifying which functions to consider for the tests, and as output it produces intermediate results in JSON format. [The second part](#part-2) takes as input the JSONs produced by the first part, and it produces as output four CSVs with the raw FunctionSimSearch hashes for the four tested configurations. If you want evaluation files such as `*_sim.csv`, there is one more conversion step using the existing pairs files in `DBs/.../pairs` ([Part 3](#part-3)).

This tool is based on this project by Thomas Dullien: https://github.com/googleprojectzero/functionsimsearch. We forked the repository (commit `ec5d9e1224ff915b3dc9e7af19e21e110c25239c`) and we customized it to our needs, integrated our layers of analysis, and tweaked the docker container. The specific changes to the initial Google P0 repository are documented in the file `functionsimsearch.patch`.


## Part 1
Before running the IDA plugin, make sure `click` is available in the Python
environment that launches `cli_fss.py`.

If you use `-c/--use-capstone`, the `capstone` Python module must also be
available inside IDA's own Python environment. Without it, the command fails
explicitly instead of silently generating empty results.

- **Input**: the JSON file with the selected functions (`-j`), the output
  directory (`-o`), and (`-c`) to use Capstone to disassemble.
- **Output**: one JSON file per IDB

Useful options:
- `--ida-path`: path to `idat`/`idat64`, or the IDA installation directory
- `-n/--jobs`: number of IDA processes to run in parallel (`0` means auto)
- `--log-dir`: where per-IDB IDA logs are written (default:
  `<output-dir>/logs`)

**Note**: the path of the IDB files in the JSON in input **must be relative** to the `binary_function_similarity` directory. The Python3 script converts the relative path into a full path to correctly load the IDB in IDA Pro.

Example: run the plugins over the functions selected for the Dataset-Vulnerability test (requires the IDBs in the `IDBs/Dataset-Vulnerability` directory)
```bash
cd IDA_fss
python3 cli_fss.py \
  --ida-path /home/damaoooo/ida-pro-9.3 \
  -j ../../../DBs/Dataset-Vulnerability/features/selected_Dataset-Vulnerability.json \
  -o fss_Dataset-Vulnerability \
  -n 16
```

Dataset-1 testing:
```bash
cd Models/functionsimsearch/IDA_fss
python3 cli_fss.py \
  --ida-path /home/damaoooo/ida-pro-9.3 \
  -j ../../../DBs/Dataset-1/features/testing/selected_testing_Dataset-1.json \
  -o ../fss_Dataset-1-testing \
  -n 16
```

Dataset-2 testing:
```bash
cd Models/functionsimsearch/IDA_fss
python3 cli_fss.py \
  --ida-path /home/damaoooo/ida-pro-9.3 \
  -j ../../../DBs/Dataset-2/features/selected_testing_Dataset-2.json \
  -o ../fss_Dataset-2 \
  -n 16
```

Run unit tests:
```bash
python3 -m unittest test_fss.py

# optional - test on multiple files
python3 -m unittest test_large_fss.py
```


## Part 2
- **Input**: a directory with JSONs obtained via the IDA_fss plugin
- **Output**: four CSVs, each of which containing the raw hashes for one of the specific four tested configurations

Useful options:
- `-n/--jobs`: number of worker processes used to hash input JSON files in
  parallel (`0` means auto)
- `-v/--verbose`: print per-input JSON completion details

The parallelization is file-based: one worker processes one input JSON at a
time. This scales well when the input directory contains many `_fss.json`
files. If the workload is dominated by one very large JSON, further sharding
would need to happen at the function level.

Example input: [testdata/fss_jsons](testdata/fss_jsons).
Example output: [testdata/fss_csvs](testdata/fss_csvs).

### Tested configurations:
The following are the four configurations we tested. In each test we assign different weights to each feature.

| config | immediate | mnemonic | graphlet |
|--------|-----------|----------|----------|
|      1 |      4.00 |     0.05 |     1.00 |
|      2 |      0.00 |     0.00 |     1.00 |
|      3 |      0.00 |     1.00 |     1.00 |
|      4 |      1.00 |     1.00 |     1.00 |


### Build and run the Docker container
These are the concrete steps to run the analysis within the provided Docker container:

- Clone the functionsimsearch repository and apply the patch:
```bash
git clone https://github.com/googleprojectzero/functionsimsearch;
( cd functionsimsearch ; git checkout ec5d9e1224ff915b3dc9e7af19e21e110c25239c ; patch -s -p0 < ../functionsimsearch.patch );
cp fss_simhasher.py ./functionsimsearch/;
```

- Build the docker image:
```bash
docker build -t fss ./functionsimsearch
```

- Run the main script within the docker container: 
```bash
docker run --rm -it fss -v <full-path-to-the-input-jsons-dir>:/input -v <full-path-to-the-output-csvs-dir>:/output /fss_simhasher.py -n 16
```

Example (it creates four CSVs in `/tmp/fss_csvs`):
```bash
docker run --rm -v $(pwd)/testdata/fss_jsons:/input -v /tmp/fss_csvs:/output -it fss /fss_simhasher.py
```

- Run the script for Dataset-1 testing
```bash
cd Models/functionsimsearch
docker run --rm \
  -v "$(pwd)/fss_Dataset-1-testing":/input \
  -v "$(pwd)/../../Results/FunctionSimSearch/Dataset-1-testing":/output \
  -it fss /fss_simhasher.py -n 16
```

- Run the script for Dataset-2
```bash
cd Models/functionsimsearch
docker run --rm \
  -v "$(pwd)/fss_Dataset-2":/input \
  -v "$(pwd)/../../Results/FunctionSimSearch/Dataset-2":/output \
  -it fss /fss_simhasher.py -n 16
```

This produces these four files in the output directory:
- `IMM:4.00_MNEM:0.05_GRAPH:1.00.csv`
- `IMM:0.00_MNEM:0.00_GRAPH:1.00.csv`
- `IMM:0.00_MNEM:1.00_GRAPH:1.00.csv`
- `IMM:1.00_MNEM:1.00_GRAPH:1.00.csv`

These are raw FunctionSimSearch outputs, not final `*_sim.csv` evaluation files.

## Part 3
To compute recall@K or other pairwise metrics, merge the raw FSS CSVs from Part 2 with the existing pairs files:

- Dataset-1: `DBs/Dataset-1/pairs/testing`
- Dataset-2: `DBs/Dataset-2/pairs`
- Dataset-Vulnerability: `DBs/Dataset-Vulnerability/pairs`

The repository now includes a CLI replacement for the notebook:
- [`convert_fss_results.py`](convert_fss_results.py)

It reproduces the notebook logic, but:
- takes the raw FSS input and output directory as command-line arguments
- takes the pairs CSV directory as a command-line argument
- when the FSS input is a directory, it automatically selects only
  `IMM:0.00_MNEM:0.00_GRAPH:1.00.csv`
- shows a progress bar
- can parallelize the conversion across multiple output files
- writes repository-style `*_sim.csv` files such as
  `pos_testing_Dataset-1_sim.csv`

Dataset-1 example:
```bash
cd Models/functionsimsearch
python3 convert_fss_results.py \
  -i ../../Results/FunctionSimSearch/Dataset-1-testing \
  -p ../../DBs/Dataset-1/pairs/testing \
  -o ../../Results/FunctionSimSearch/Dataset-1-testing-sim \
  -n 4
```

Dataset-2 example:
```bash
cd Models/functionsimsearch
python3 convert_fss_results.py \
  -i /path/to/raw_fss/Dataset-2 \
  -p /path/to/DBs/Dataset-2/pairs \
  -o /path/to/output/Dataset-2 \
  -n 4
```

Custom pairs directory example:
```bash
cd Models/functionsimsearch
python3 convert_fss_results.py \
  -i /path/to/raw_fss_csvs \
  -p /home/damaoooo/Downloads/binary_function_similarity/DBs/Dataset-1/pairs/testing \
  -o /path/to/output \
  -n 4
```

The original notebook is still useful as a reference:
- [`Results/notebooks/Convert FunctionSimSearch results.ipynb`](../../Results/notebooks/Convert%20FunctionSimSearch%20results.ipynb)

In other words, Part 2 gives you the hashes, and Part 3 converts them into pairwise similarity CSVs.

## Copyright information about FunctionSimSearch

[FunctionSimSearch](https://github.com/googleprojectzero/functionsimsearch) is released under Apache License 2.0.

[IDA_fss.py](IDA_fss/IDA_fss.py) includes part of the code from https://github.com/williballenthin/python-idb/ and https://github.com/googleprojectzero/functionsimsearch which are licensed under Apache License 2.0.
