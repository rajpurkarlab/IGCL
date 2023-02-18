"""
Make candidate sets for query images in image-image retrieval tasks. The result 
of running this script will be handled to a radiologist for further annotation.

Abnormalities categories include (8 total):
Fracture, Edema, Pneumonia, Cardiomegaly, Pneumothorax, Atelectasis, Pleural Effusion, No Finding
"""

from collections import defaultdict
import argparse
import pandas as pd
import random

random.seed(1234)

ALL_VARS = ["Fracture", "Edema", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Atelectasis", "Pleural Effusion", "No Finding"]
START_IDX = 5 # starting index of all variables in CheXpert file headers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("chexpert_csv_file", type=str)
    parser.add_argument("output_candidate_file", type=str)
    parser.add_argument("--n_candidate", type=int, default=50, help="Number of candidate images to keep for each variable.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # read chexpert csv
    df = pd.read_csv(args.chexpert_csv_file)
    df = df.fillna(0)
    # keep only frontal images for simplicity
    df = df[df['Frontal/Lateral'] == 'Frontal']
    total = len(df)
    print(f"{total} frontal image examples loaded from file: {args.chexpert_csv_file}")

    # get matrix of all variables
    all_headers = list(df.columns.values)
    all_headers = all_headers[START_IDX:]
    # map all -1 to 2 for easier aggregating of results
    df = df.replace({-1:2})
    matrix = df[all_headers].to_numpy()

    var2candidates = dict()
    # for all variables, get query row indices and candidate row indices in the df file
    for var in ALL_VARS:
        col_idx = all_headers.index(var)
        row_idxs = get_exclusive_positive_row_indices(col_idx, matrix)
        print(f"{len(row_idxs)} exclusively positive examples found for var: {var}")
        # make sure we have enough images
        assert len(row_idxs) >= args.n_candidate, \
            f"Cannot find enough images for var: {var}"
        random.shuffle(row_idxs)

        # queries
        candidates = row_idxs[:args.n_candidate]
        cand_paths = df['Path'].iloc[candidates].tolist()
        assert len(cand_paths) == args.n_candidate
        var2candidates[var] = cand_paths

    # save candidate paths to file
    # make dictionary that maps column names to lists
    column2list = defaultdict(list)
    for var, cands in var2candidates.items():
        var_list = [var] * len(cands)
        column2list['Variable'] += var_list
        column2list['Path'] += cands
    cand_df = pd.DataFrame(data=column2list)
    cand_df.to_csv(args.output_candidate_file, index=False)
    print(f"{len(cand_df)} total candidate image paths saved to file: {args.output_candidate_file}")


def get_exclusive_positive_row_indices(col_idx, matrix):
    """
    Matrix should be filled with values of 0, 1 or 2
    """
    ncol = matrix.shape[1]
    col = matrix[:, col_idx]
    other_idxs = [i for i in range(ncol) if i != col_idx]
    other_cols = matrix[:, other_idxs]
    assert len(col) == len(other_cols)

    rows = []
    for i in range(len(col)):
        other_vals = other_cols[i]
        if col[i] == 1 and other_vals.sum() == 0: # only current is positive
            rows.append(i)
    return rows


if __name__ == "__main__":
    main()