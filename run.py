import diversity
import argparse
import pandas as pd


# FIXME doesnt require pandas dependency, could be done using only numpy (would have to remove header)
def process_input_file(filepath):
    df = pd.read_csv(filepath, header=None)
    column_names = ['species', 'count', 'community']
    n_features = df.shape[1] - len(column_names)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    column_names += feature_names
    df.columns = column_names
    # FIXME 'species' and 'community' are currently unused
    features = df.iloc[:, 3:].values
    counts = df['count'].values
    return features, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        type=str,
        help="A list of csv file(s) where the first two columns of the file(s) are the species name and its count and all following columns are features of that species that will be used to calculate similarity between species")
    parser.add_argument(
        "-q",
        nargs='+',
        type=float,
        help="A list of q's where each q >= 0")
    parser.add_argument(
        "--similarity_matrix",
        type=str,
        help="A filepath the csv file containing a symmetric similarity matrix")

    args = parser.parse_args()
    features, counts = process_input_file(args.filepath)

    qDs = [diversity.alpha(features, counts, q, filepath=args.similarity_matrix)
           for q in args.q]

    df = pd.DataFrame({'q': args.q, 'qDs': qDs})
    print(df)


if __name__ == "__main__":
    main()
