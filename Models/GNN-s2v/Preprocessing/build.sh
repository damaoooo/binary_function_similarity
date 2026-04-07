source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

set -e

REPO=/home/damaoooo/Downloads/binary_function_similarity
export IDA_PATH=/home/damaoooo/ida-pro-9.3/idat
NPROC=32

cd "$REPO"

# 4) 生成 GNN-s2v 的 Dataset-1 预处理产物
for split in training validation testing; do
  OUTDIR="Models/GNN-s2v/Preprocessing/Dataset-1_${split}"
  mkdir -p "$OUTDIR"

  python Models/GNN-s2v/Preprocessing/digraph_instructions_embeddings.py \
    -i "DBs/Dataset-1/features/${split}/acfg_disasm_Dataset-1_${split}" \
    -d "Models/GNN-s2v/Pretraining/Dataset-1_training/ins2id.json" \
    -p "$NPROC" \
    -o "$OUTDIR"

  python Models/GNN-s2v/Preprocessing/digraph_numerical_features.py \
    -i "DBs/Dataset-1/features/${split}/acfg_features_Dataset-1_${split}" \
    -p "$NPROC" \
    -o "$OUTDIR"
done

# 5) 生成 GNN-s2v 的 Dataset-2 预处理产物
mkdir -p "Models/GNN-s2v/Preprocessing/Dataset-2"

python Models/GNN-s2v/Preprocessing/digraph_instructions_embeddings.py \
  -i "DBs/Dataset-2/features/acfg_disasm_Dataset-2" \
  -d "Models/GNN-s2v/Pretraining/Dataset-1_training/ins2id.json" \
  -p "$NPROC" \
  -o "Models/GNN-s2v/Preprocessing/Dataset-2"

python Models/GNN-s2v/Preprocessing/digraph_numerical_features.py \
  -i "DBs/Dataset-2/features/acfg_features_Dataset-2" \
  -p "$NPROC" \
  -o "Models/GNN-s2v/Preprocessing/Dataset-2"
