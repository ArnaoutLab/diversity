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
    sumcommunity_names = list(df['subcommunity'].unique()) * len(args.q)
    Z_filepath = args.Z
    alpha = [diversity.raw_alpha(df, q, z_filepath=Z_filepath)
             for q in args.q]
    alpha_bar = [diversity.normalized_alpha(df, q, z_filepath=Z_filepath)
                 for q in args.q]
    df = pd.DataFrame({'q': args.q, 'raw_alpha': alpha,
                      'normalized_alpha': alpha_bar})
    df = df.explode(['raw_alpha', 'normalized_alpha'], ignore_index=True)
    df.insert(1, "subcommunity", sumcommunity_names)
    print(df)  # FIXME delete me


if __name__ == "__main__":
    main()
