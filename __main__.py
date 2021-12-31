from argparse import ArgumentParser
from logging import captureWarnings
from pathlib import Path
from platform import python_version
from sys import argv
from warnings import warn

from numpy import (
    genfromtxt,
    repeat,
    )
from pandas import (
    DataFrame,
    read_csv,
    )

from Chubacabra.diversity import Metacommunity
from Chubacabra.log import (
    LOG_HANDLER,
    LOGGER)
from Chubacabra.parameters import configure_arguments

# Ensure warnings are handled properly.
captureWarnings(True)
getLogger('py.warnings').addHandler(LOG_HANDLER)

########################################################################
def main():
    # Parse and validate arguments
    parser = configure_arguments()
    args = parser.parse_args()

    LOGGER.setLevel(args.log_level)
    LOGGER.info(' '.join([f'python{python_version()}', *argv]))
    LOGGER.debug(f'args: {args}')

    # FIXME equal species are listed multiple times (once for each
    # subcommunity they are members of, which is deceiving 
    data = genfromtxt(args.filepath, delimiter=',', dtype=object)
    LOGGER.debug(f'data: {data}')
    counts = data[:, :3]
    features = data[:, 3:]
    meta = Metacommunity(counts, args.q, args.Z, features=features)

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

    LOGGER.info('Done!')

########################################################################
if __name__ == "__main__":
    main()
