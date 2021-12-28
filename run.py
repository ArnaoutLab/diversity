import numpy as np
import diversity
import argparse
import warnings
import pandas as pd
from pathlib import Path


# FIXME doesnt require pandas dependency, could be done using only numpy (would have to remove header)
def process_input_file(filepath):
    df = pd.read_csv(filepath, header=None)
    column_names = ['species', 'count', 'subcommunity']
    n_features = df.shape[1] - len(column_names)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    column_names += feature_names
    df.columns = column_names
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        type=str,
        help="A csv file where the first 3 columns of the file are the species name, its count, and subcommunity name, and all following columns are features of that species that will be used to calculate similarity between species")
    parser.add_argument(
        "Z",
        type=str,
        help="The filepath to a csv file containing a symmetric similarity matrix. If the file does not exist, one will be created with the user defined similarity function")
    parser.add_argument(
        "-q",
        nargs='+',
        type=float,
        help="A list of q's where each q >= 0")

    args = parser.parse_args()

    # validate arguments
    if any([q > 100 for q in args.q]):
        warnings.warn(
            "q > 100.0 defaults to the analytical formula for q = inf", stacklevel=2)

    df = process_input_file(args.filepath)
    print(df)  # FIXME delete me

    sumcommunity_names = list(df['subcommunity'].unique())
    sumcommunity_column = sumcommunity_names * len(args.q)
    q_column = np.repeat(args.q, len(sumcommunity_names))
    Z_filepath = args.Z

    fs = [diversity.raw_alpha, diversity.normalized_alpha,
          diversity.raw_rho, diversity.normalized_rho, diversity.gamma]

    measures = [[f(df, q, z_filepath=Z_filepath) for f in fs] for q in args.q]

    columns = ['raw_alpha', 'normalized_alpha',
               'raw_rho', 'normalized_rho', 'gamma']
    df = pd.DataFrame(measures)
    df.columns = columns

    df = df.explode(columns[:-1], ignore_index=True)
    df = df.explode(['gamma'])
    df.insert(0, "subcommunity", sumcommunity_column)
    df.insert(0, "q", q_column)
    print(df)  # FIXME delete me


if __name__ == "__main__":
    main()
