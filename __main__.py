from argparse import ArgumentParser
from logging import captureWarnings
from pathlib import Path
from platform import python_version
from sys import argv
from warnings import warn

from numpy import repeat
from pandas import (
    DataFrame,
    read_csv,
    )

from Chubacabra.diversity import (
    gamma,
    normalized_alpha,
    normalized_rho,
    raw_alpha,
    raw_rho,
    )
from Chubacabra.log import (
    LOG_HANDLER,
    LOGGER)
from Chubacabra.parameters import configure_arguments

# Ensure warnings are handled properly.
captureWarnings(True)
getLogger('py.warnings').addHandler(LOG_HANDLER)

# FIXME doesnt require pandas dependency, could be done using only numpy (would have to remove header)
def process_input_file(filepath):
    df = read_csv(filepath, header=None)
    column_names = ['species', 'count', 'subcommunity']
    n_features = df.shape[1] - len(column_names)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    column_names += feature_names
    df.columns = column_names
    return df

########################################################################
def main():
    # Parse and validate arguments
    parser = configure_arguments()
    args = parser.parse_args()

    LOGGER.setLevel(args.log_level)
    LOGGER.info(' '.join([f'python{python_version()}', *argv]))
    LOGGER.debug(f'args: {args}')

    df = process_input_file(args.filepath)
    LOGGER.debug(f'df: {df}')

    subcommunity_names = list(df['subcommunity'].unique())
    subcommunity_column = subcommunity_names * len(args.q)
    q_column = repeat(args.q, len(subcommunity_names))
    Z_filepath = args.Z

    fs = [raw_alpha, normalized_alpha, raw_rho, normalized_rho, gamma]

    measures = [[f(df, q, z_filepath=Z_filepath) for f in fs] for q in args.q]

    columns = ['raw_alpha', 'normalized_alpha',
               'raw_rho', 'normalized_rho', 'gamma']
    df = DataFrame(measures)
    df.columns = columns

    df = df.explode(columns[:-1], ignore_index=True)
    df = df.explode(['gamma'])
    df.insert(0, "subcommunity", subcommunity_column)
    df.insert(0, "q", q_column)
    LOGGER.debug(f'df: {df}')

    LOGGER.info('Done!')

########################################################################
if __name__ == "__main__":
    main()
