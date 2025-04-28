cd /home/damaoooo/Downloads/binary_function_similarity/IDA_scripts

cd IDA_acfg_disasm
/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_disasm.py -j ../../DBs/Dataset-1/features/testing/selected_testing_Dataset-1.json -o ../../DBs/Dataset-1/features/testing/acfg_disasm_testing_Dataset-1
/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_disasm.py -j ../../DBs/Dataset-1/features/training/selected_training_Dataset-1.json -o ../../DBs/Dataset-1/features/training/acfg_disasm_training_Dataset-1
/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_disasm.py -j ../../DBs/Dataset-1/features/validation/selected_validation_Dataset-1.json -o ../../DBs/Dataset-1/features/validation/acfg_disasm_validation_Dataset-1

/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_disasm.py -j ../../DBs/Dataset-2/features/selected_testing_Dataset-2.json -o ../../DBs/Dataset-2/features/acfg_disasm_testing_Dataset-2


cd /home/damaoooo/Downloads/binary_function_similarity/IDA_scripts
cd IDA_acfg_features

/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_features.py -j ../../DBs/Dataset-1/features/testing/selected_testing_Dataset-1.json -o ../../DBs/Dataset-1/features/testing/acfg_features_testing_Dataset-1
/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_features.py -j ../../DBs/Dataset-1/features/training/selected_training_Dataset-1.json -o ../../DBs/Dataset-1/features/training/acfg_features_training_Dataset-1
/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_features.py -j ../../DBs/Dataset-1/features/validation/selected_validation_Dataset-1.json -o ../../DBs/Dataset-1/features/validation/acfg_features_validation_Dataset-1
/home/damaoooo/miniconda3/envs/ml/bin/python cli_acfg_features.py -j ../../DBs/Dataset-2/features/selected_testing_Dataset-2.json -o ../../DBs/Dataset-2/features/acfg_features_testing_Dataset-2

