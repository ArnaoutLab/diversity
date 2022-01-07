from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv, concat

from metacommunity.diversity import Metacommunity
from metacommunity.parameters import configure_arguments
from metacommunity.log import LOG_HANDLER, LOGGER

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

    species_counts = read_csv(args.input_file)

    LOGGER.debug(f'data: {species_counts}')

    features = 'FIXME'  # FIXME read features in separately
    viewpoint = args.viewpoint

    meta = Metacommunity(species_counts, args.similarity_matrix_file)

    subcommunity_views = concat([meta.subcommunities_to_dataframe(view)
                                 for view in viewpoint])
    metacommunity_views = concat([meta.metacommunity_to_dataframe(view)
                                  for view in viewpoint])
    metacommunity_views.columns = subcommunity_views.columns
    community_views = concat(
        [subcommunity_views, metacommunity_views])
    community_views.to_csv(args.output_file, sep='\t',
                           float_format='%.2f', index=False)

    LOGGER.info('Done!')


########################################################################
if __name__ == "__main__":
    main()
