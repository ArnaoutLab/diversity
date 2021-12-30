import numpy as np
import diversity
import argparse
import warnings


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

    data = np.genfromtxt(args.filepath, delimiter=',', dtype=object)
    print(data)  # FIXME delete me
    counts = data[:, :3]
    features = data[:, 3:]
    meta = diversity.Metacommunity(counts, args.q, args.Z, features=features)

    print('\n')

    print(meta.raw_alpha)
    print(meta.normalized_alpha)
    print(meta.raw_rho)
    print(meta.normalized_rho)
    print(meta.raw_beta)
    print(meta.normalized_beta)
    print(meta.gamma)

    print('\n')

    print(meta.A)
    print(meta.normalized_A)
    print(meta.R)
    print(meta.normalized_R)
    print(meta.B)
    print(meta.normalized_B)
    print(meta.G)


if __name__ == "__main__":
    main()
