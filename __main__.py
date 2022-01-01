from parameters import configure_arguments
from log import LOG_HANDLER, LOGGER
from diversity import Metacommunity
from numpy import genfromtxt
from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger


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

    print(meta.alpha)
    print(meta.normalized_alpha)
    print(meta.rho)
    print(meta.normalized_rho)
    print(meta.beta)
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
