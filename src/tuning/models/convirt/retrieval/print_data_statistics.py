"""
Print data statistics from CheXpert csv files.
"""

import pandas as pd

IN_FN = 'dataset/chexpert/valid.csv'
# IN_FN = 'dataset/chexpert/train.csv'
# the starting index of meaningful headers
START_IDX = 5

def main():
    # load csv
    df = pd.read_csv(IN_FN)
    df = df.fillna(0)
    headers = list(df.columns.values)
    headers = headers[START_IDX:]
    total = len(df)
    print(f"{total} examples loaded from file.")
    print(f"All headers: {headers}")

    # map all -1 to 2 for easier aggregating of results
    df = df.replace({-1:2})
    matrix = df[headers].to_numpy()

    # print basic binary value statistics
    print("\nStatistics of all fields:")
    for h in headers:
        vals = df[h].tolist()
        total_pos = len([x for x in vals if x == 1])
        ratio_pos = total_pos / total * 100
        print(f"{h}:\t\t{total_pos}\t{ratio_pos:.1f}%")
    
    # print exclusive positivity statistics
    # i.e., for each variable, find cases where only this variable is positive
    print("\nStatistics of exclusive positivity:")
    for idx, h in enumerate(headers):
        exclusive_pos = count_exclusive_positive_values(idx, matrix)
        ratio_expos = exclusive_pos / total * 100
        print(f"{h}:\t\t{exclusive_pos}\t{ratio_expos:.1f}%")


def count_exclusive_positive_values(col_idx, matrix):
    """
    Matrix should be filled with values of 0, 1 or 2
    """
    ncol = matrix.shape[1]
    col = matrix[:, col_idx]
    other_idxs = [i for i in range(ncol) if i != col_idx]
    other_cols = matrix[:, other_idxs]
    assert len(col) == len(other_cols)

    count = 0
    for i in range(len(col)):
        other_vals = other_cols[i]
        if col[i] == 1 and other_vals.sum() == 0: # only current is positive
            count += 1
    return count

if __name__ == "__main__":
    main()