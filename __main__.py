from parameters import configure_arguments
from log import LOG_HANDLER, LOGGER
from diversity import Metacommunity
from numpy import genfromtxt
from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv

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
    counts = data.iloc[:,:3].to_numpy()
    unique_species_correspondence = UniqueRowsCorrespondence(counts, 0)
    unique_species = unique_species_correspondence.unique_keys
    unique_species_rows = unique_species_correspondence.unique_row_index
    features = data.iloc[:,3:][unique_species_rows]
    viewpoint = args.q[0]
    meta = Metacommunity(counts, unique_species_correspondence,
                         unique_species, viewpoint, args.Z,
                         features=features)

    print('\n')

    print(meta.alpha)
    print(meta.rho)
    print(meta.beta)
    print(meta.gamma)
    print(meta.normalized_alpha)
    print(meta.normalized_rho)
    print(meta.normalized_beta)

    print('\n')

    print(meta.A)
    print(meta.R)
    print(meta.B)
    print(meta.G)
    print(meta.normalized_A)
    print(meta.normalized_R)
    print(meta.normalized_B)

    LOGGER.info('Done!')


########################################################################
if __name__ == "__main__":
    main()
