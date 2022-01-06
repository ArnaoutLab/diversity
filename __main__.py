from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv

from diversity import Metacommunity
from parameters import configure_arguments
from log import LOG_HANDLER, LOGGER

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

    species_counts = read_csv(args.filepath).to_numpy()

    LOGGER.debug(f'data: {species_counts}')

    features = 'FIXME'  # FIXME read features in separately
    viewpoint = args.viewpoint[0]

    meta = Metacommunity(species_counts, args.similarities)

    print('\n')
    print(meta.subcommunities_to_dataframe(viewpoint))
    print('\n')
    print(meta.metacommunity_to_dataframe(viewpoint))
    print('\n')

    LOGGER.info('Done!')


########################################################################
if __name__ == "__main__":
    main()
