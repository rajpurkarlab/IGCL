"""
Make folders that contain candidate query images. All images for an abnormality
will be copied into an individual folder for annotation purpose.

All folders will be uploaded to MD.ai for further annotations.
"""

from collections import defaultdict
import argparse
import pandas as pd
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str)
    parser.add_argument("target_dir", type=str)
    parser.add_argument("candidate_file", type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    df = pd.read_csv(args.candidate_file)
    var2candidates = defaultdict(list)
    for index, row in df.iterrows():
        var = row['Variable']
        path = row['Path']
        var2candidates[var].append(path)

    # go through images and make folders
    if not os.path.exists(args.target_dir):
        print(f"Creating directory {args.target_dir}...")
        os.makedirs(args.target_dir)
    print(f"Copying all candidate files from dir {args.image_dir} to dir {args.target_dir}...")
    for var, cand_list in var2candidates.items():
        var_dir = os.path.join(args.target_dir, var.lower().replace(' ', '_'))
        if not os.path.exists(var_dir):
            print(f"Creating directory {var_dir}...")
            os.makedirs(var_dir)
        print(f"Copying images for variable: {var} to {var_dir}")
        for cand_path in cand_list:
            short_path = cand_path.split('/', 1)[1]
            src_path = os.path.join(args.image_dir, short_path)
            filename = short_path.replace('/', '-')
            tgt_path = os.path.join(var_dir, filename)
            # copy file
            shutil.copyfile(src_path, tgt_path)

    print("Done!")

if __name__ == "__main__":
    main()