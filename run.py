import numpy as np
import diversity
import argparse
import warnings
import pandas as pd


# FIXME not sure where this function should go
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
    z_filepath = args.Z
    meta = diversity.Metacommunity(df, z_filepath=z_filepath)
    fs = [meta.raw_alpha, meta.normalized_alpha,
          meta.raw_rho, meta.normalized_rho, meta.gamma]
    measures = [[f(q) for q in args.q] for f in fs]
    columns = ['raw_alpha', 'normalized_alpha',
               'raw_rho', 'normalized_rho', 'gamma']
    df = pd.DataFrame(measures).T
    df.columns = columns
    df = df.explode(columns[:-1], ignore_index=True)
    df = df.explode(['gamma'])
    df.insert(0, "subcommunity", sumcommunity_column)
    df.insert(0, "q", q_column)
    df.insert(4, 'raw_beta', 1 / df['raw_rho'])
    df.insert(5, 'normalized_beta', 1 / df['normalized_rho'])
    print(df)  # FIXME delete me


if __name__ == "__main__":
    main()
