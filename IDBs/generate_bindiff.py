import os
import subprocess
import multiprocessing
from tqdm import tqdm

def execute_bindiff(primary, secondary, output_path):
    commands = ["bindiff", "--primary", primary, "--secondary", secondary, "--output_dir", output_path]
    sub_process = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, _ = sub_process.communicate()
    return 

def generate_bindiff(unstrip_folder: str, strip_folder: str, output_folder: str):
    
    # if output folder does not exist, create it
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        
    # Clear the output folder
    for root, _, files in os.walk(output_folder):
        for fname in files:
            os.remove(os.path.join(root, fname))
        
    command_pool = []
    
    for sub_folder in os.listdir(unstrip_folder):
        unstrip_path = os.path.join(unstrip_folder, sub_folder)
        strip_path = os.path.join(strip_folder, sub_folder)
        if not os.path.isdir(unstrip_path) or not os.path.isdir(strip_path):
            continue
        
        # if subfolder does not exist in the output folder, create it
        output_subfolder = os.path.join(output_folder, sub_folder)
        if not os.path.isdir(output_subfolder):
            os.mkdir(output_subfolder)
        
        for root, _, files in os.walk(unstrip_path):
            for fname in files:
                if fname.endswith(".BinExport"):
                    base_name = fname.replace(".BinExport", "")
                    unstrip_binexport = os.path.join(unstrip_path, fname)
                    strip_binexport = os.path.join(strip_path, fname)
                    
                    command_pool.append((unstrip_binexport, strip_binexport, output_subfolder))
                    
    pool = multiprocessing.Pool(processes=14)
    
    bar = tqdm(total=len(command_pool), desc="Generating BinDiff")
    for cmd in command_pool:
        p = pool.apply_async(execute_bindiff, args=cmd, callback=lambda _: bar.update(1))
        
    pool.close()
    pool.join()
    bar.close()
    

if __name__ == '__main__':
    unstrip_path = "/home/damaoooo/Downloads/binary_function_similarity/IDBs/Dataset-1"
    strip_path = "/home/damaoooo/Downloads/binary_function_similarity/IDBs/Dataset-1-strip"
    output_path = "/home/damaoooo/Downloads/binary_function_similarity/IDBs/BinDiffOutput"
    generate_bindiff(unstrip_path, strip_path, output_path)  