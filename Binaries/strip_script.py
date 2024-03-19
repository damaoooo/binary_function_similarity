import os
import multiprocessing
import subprocess
from tqdm import tqdm



def get_stripper_path(dir_path, arch_name):
    if arch_name == "arm32":
        return os.path.join(dir_path, "strip_arm")
    elif arch_name == "mips32":
        return os.path.join(dir_path, "strip_mips")
    elif arch_name == "arm64":
        return os.path.join(dir_path, "strip_aarch64")
    elif arch_name == "mips64":
        return os.path.join(dir_path, "strip_mips64")
    else:
        return "strip"

def strip_file_command(file_path, stripper_path):
    path, file = os.path.split(file_path)
    if file.startswith("arm32"):
        commands = ["qemu-arm-static", get_stripper_path(stripper_path, "arm32"), "-s", "-D", file_path]
    elif file.startswith("mips32"):
        commands = ["qemu-mips-static", get_stripper_path(stripper_path, "mips32"), "-s", "-D", file_path]
    elif file.startswith("arm64"):
        commands = ["qemu-aarch64-static", get_stripper_path(stripper_path, "arm64"), "-s", "-D", file_path]
    elif file.startswith("mips64"):
        commands = ["qemu-mips64-static", get_stripper_path(stripper_path, "mips64"), "-s", "-D", file_path]
    else:
        commands = ["strip", file_path]
    return commands

def execute_commands(cmds):
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err:
        print(err)
    return out, err

def main():
    
    stripper_path = "Binaries/Strip_Binaries"
    binary_path = "Dataset-1-strip"
    
    commands = []

    for root, dirs, files in os.walk(binary_path):
        for file in files:
            file_path = os.path.join(root, file)
            commands.append(strip_file_command(file_path, stripper_path))
            
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    bar = tqdm(total=len(commands), desc="Stripping files")
    
    for cmd in commands:
        pool.apply_async(execute_commands, args=(cmd,), callback=lambda _: bar.update(1))
        
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main()
            
    
    