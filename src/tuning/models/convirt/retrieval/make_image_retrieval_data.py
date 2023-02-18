"""
Make an image retrieval evaluation set from the CheXpert training set.

Abnormalities categories include:
Fracture, Edema, Pneumonia, Cardiomegaly, Pneumothorax, Atelectasis, Pleural Effusion

Normal categories are taken from images with "No Finding".
"""

from collections import defaultdict
import argparse
import pandas as pd
import random

random.seed(1234)

# ALL_VARS = ["Fracture", "Edema", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Atelectasis", "Pleural Effusion"]
ALL_VARS = ["Fracture", "Edema", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Atelectasis", "Pleural Effusion", "No Finding"]
NORMAL_VAR = 'No Finding'
START_IDX = 5 # starting index of all variables in CheXpert file headers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("chexpert_csv_file", type=str)
    parser.add_argument("output_data_file", type=str)
    parser.add_argument("output_query_file", type=str)
    parser.add_argument("--n_query", type=int, default=10, help="Number of query images to use for each variable.")
    parser.add_argument("--n_candidate", type=int, default=100, help="Number of candidate images to keep for each variable.")
    parser.add_argument("--n_normal", type=int, default=2000, help="Number of normal images to keep.")
    
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

    var2queries = dict()
    all_candidate_idxs = []
    # for all variables, get query row indices and candidate row indices in the df file
    for var in ALL_VARS:
        col_idx = all_headers.index(var)
        row_idxs = get_exclusive_positive_row_indices(col_idx, matrix)
        print(f"{len(row_idxs)} exclusively positive examples found for var: {var}")
        # make sure we have enough images
        assert len(row_idxs) >= args.n_query + args.n_candidate, \
            f"Cannot find enough images for var: {var}"
        random.shuffle(row_idxs)

        # queries
        queries = row_idxs[:args.n_query]
        query_paths = df['Path'].iloc[queries].tolist()
        assert len(query_paths) == args.n_query
        var2queries[var] = query_paths

        candidates = row_idxs[args.n_query:args.n_query+args.n_candidate]
        # candidates = row_idxs[:args.n_candidate]
        all_candidate_idxs += candidates
    
    # find normal examples
    if args.n_normal > 0:
        col_idx = all_headers.index(NORMAL_VAR)
        row_idxs = get_exclusive_positive_row_indices(col_idx, matrix)
        print(f"{len(row_idxs)} normal examples found.")
        assert len(row_idxs) >= args.n_normal
        random.shuffle(row_idxs)
        all_candidate_idxs += row_idxs[:args.n_normal]

    # save examples to file
    df_to_save = df.iloc[all_candidate_idxs]
    df_to_save.to_csv(args.output_data_file, index=False)
    print(f"{len(df_to_save)} total examples saved to file: {args.output_data_file}")

    # save queries to file
    # make dictionary that maps column names to lists
    column2list = defaultdict(list)
    for var, queries in var2queries.items():
        var_list = [var] * len(queries)
        column2list['Variable'] += var_list
        column2list['Path'] += queries
    query_df = pd.DataFrame(data=column2list)
    query_df.to_csv(args.output_query_file, index=False)
    print(f"{len(query_df)} total queries saved to file: {args.output_query_file}")


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