from functools import partial
from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv

from diversity import Metacommunity
from utilities import UniqueRowsCorrespondence, register
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

    species_to_id = {}
    subcommunity_to_id = {}

    register_species = partial(register, registry=species_to_id)
    register_subcommunity = partial(register, registry=subcommunity_to_id)

    data = read_csv(args.filepath, comment='#',
                    converters={0: register_species, 2: register_subcommunity})

    LOGGER.debug(f'data: {data}')
    counts = data.to_numpy()
    features = 'FIXME'  # FIXME read features in separately
    viewpoint = args.q[0]
    meta = Metacommunity(counts, args.Z)

    print('\n')
    print(meta.subcommunities_to_dataframe(viewpoint))
    print('\n')
    print(meta.metacommunity_to_dataframe(viewpoint))
    print('\n')

    LOGGER.info('Done!')


########################################################################
if __name__ == "__main__":
    main()
