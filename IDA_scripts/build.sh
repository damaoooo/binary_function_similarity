source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

set -e

REPO=/home/damaoooo/Downloads/binary_function_similarity
export IDA_PATH=/home/damaoooo/ida-pro-9.1/idat
NPROC=8

cd "$REPO"

# 1) 生成 Dataset-1 / Dataset-2 的 IDBs
python IDA_scripts/generate_idbs.py --db1 --db2

# 2) 重跑 Dataset-1 的 ACFG 特征
for split in training validation testing; do
  python IDA_scripts/IDA_acfg_disasm/cli_acfg_disasm.py \
    -j "DBs/Dataset-1/features/${split}/selected_${split}_Dataset-1.json" \
    -o "DBs/Dataset-1/features/${split}/acfg_disasm_Dataset-1_${split}"

  python IDA_scripts/IDA_acfg_features/cli_acfg_features.py \
    -j "DBs/Dataset-1/features/${split}/selected_${split}_Dataset-1.json" \
    -o "DBs/Dataset-1/features/${split}/acfg_features_Dataset-1_${split}"
done

# 3) 重跑 Dataset-2 的 ACFG 特征
python IDA_scripts/IDA_acfg_disasm/cli_acfg_disasm.py \
  -j "DBs/Dataset-2/features/selected_testing_Dataset-2.json" \
  -o "DBs/Dataset-2/features/acfg_disasm_Dataset-2"

python IDA_scripts/IDA_acfg_features/cli_acfg_features.py \
  -j "DBs/Dataset-2/features/selected_testing_Dataset-2.json" \
  -o "DBs/Dataset-2/features/acfg_features_Dataset-2"
