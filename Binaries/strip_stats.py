import os
import subprocess
import multiprocessing
from collections import Counter

from tqdm import tqdm

def is_stripped(output: str):
    if "not stripped" in output:
        return False
    return True

def strip_stats_single_file(file_path: str):
    output = subprocess.check_output(f"file {file_path}", shell=True).decode("utf-8")
    return is_stripped(output), file_path

def get_arch(file_path: str):
    base_name = os.path.basename(file_path)
    arch = base_name.split("-")[0]
    return arch

def strip_stats(directory: str):
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    stripped = 0
    not_stripped = 0
    
    all_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_paths.append(file_path)
            
    bar = tqdm(total=len(all_paths), desc="Checking Files", dynamic_ncols=True)
    results = []
    for file_path in all_paths:
        results.append(pool.apply_async(strip_stats_single_file, args=(file_path,), callback=lambda _: bar.update(1)))
                       
    pool.close()
    pool.join()
    
    not_stripped_files = []
    all_archtectures = []
    
    for result in results:
        success, file_path = result.get()
        arch = get_arch(file_path)
        all_archtectures.append(arch)
        if success:
            stripped += 1
        else:
            not_stripped += 1
            not_stripped_files.append(arch)
    
    print(Counter(all_archtectures))
    print(Counter(not_stripped_files))
    
    return stripped, not_stripped

if __name__ == "__main__":
    path = "/home/damaoooo/binary_function_similarity/Binaries/Dataset-1-strip"
    stripped, not_stripped = strip_stats(path)
    print(f"Stripped: {stripped}, Not Stripped: {not_stripped}")