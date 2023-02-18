"""
Utils related to disk usage.

Running `python utils/disk.py` will copy all specified files to all machines' local disks.
"""

import os
import socket
import subprocess
from pathlib import Path

# a mapping from machine name to the largest local disk dir
machine2dir = {
    'jagupard10': 'scr2',
    'jagupard11': 'scr2',
    'jagupard12': 'scr2',
    'jagupard13': 'scr2',
    'jagupard14': 'scr3',
    'jagupard15': 'scr3',
    'jagupard16': 'scr1',
    'jagupard17': 'scr1',
    'jagupard18': 'scr1',
    'jagupard19': 'scr2', # jag19 has a smaller local disk
    'jagupard20': 'scr3',
    'jagupard21': 'scr1',
    'jagupard22': 'scr1',
    'jagupard23': 'scr1',
    'jagupard24': 'scr1',
    'jagupard25': 'scr1',
    'jagupard26': 'scr1',
    'jagupard27': 'scr1'
}

def get_local_dir_for_machine(machine_name):
    if machine_name in machine2dir:
        dirname = machine2dir[machine_name]
        dirname = os.path.join(f'/{machine_name}', dirname)
        return dirname
    return None # return none if machine cannot be found

def get_local_dir():
    """
    Get the local dir to store file based on the machine.
    """
    name = socket.gethostname().split('.')[0]
    return get_local_dir_for_machine(name)

def get_local_or_remote_dir(remote_dir, local_dir, logger):
    """
    Attempt to find a local directory on the machine for `local_dir`; otherwise
    fall back to using `remote_dir`.
    """
    local_root = get_local_dir()
    target_dir = os.path.join(local_root, local_dir) if local_root else None
    if target_dir and os.path.exists(target_dir):
        logger.info(f"Using local disk dir: {target_dir}")
    else:
        # switch back to remote juice disk
        logger.warning("Cannot find local disk for this machine. Using juice disk...")
        target_dir = remote_dir
    return target_dir

def copy_files_to_machine(src_dir, tgt_dir, machine_name):
    """
    Copy files from a source directory to a target directory
    on the local disk of a machine.
    """
    local_dir = get_local_dir_for_machine(machine_name)
    if local_dir is None:
        print(f"Warning: Cannot find local disk for machine: {machine_name}. Skipping.")
    tgt_dir = os.path.join(local_dir, tgt_dir)

    # make sure parent dir exists before copying
    Path(tgt_dir).parent.mkdir(parents=True, exist_ok=True)

    cmd = f"time rsync -rptgozL --info=progress2 {src_dir} {tgt_dir}"
    print(f"Copying from {src_dir} to {tgt_dir}...")
    print(f"    with command: {cmd}")
    subprocess.run(cmd, shell=True)
    print("Done.")

def copy_files_to_all_machines(src_dir, tgt_dir, machine_list=[]):
    if len(machine_list) == 0:
        machine_list = list(machine2dir.keys())
    for machine in machine_list:
        copy_files_to_machine(src_dir, tgt_dir, machine)

if __name__ == '__main__':
    copy_files_to_all_machines('dataset/COVIDx', 'zyh/')
