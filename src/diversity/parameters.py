"""Module for Metacommunity's argument parser configuration.

Classes
-------
ArgumentWarning
    Warning category for problematic arguments.
ValidateQ
    Validator for -q parameter.

Functions
---------
configure_arguments
    Creates argument parser.
"""
from argparse import Action, ArgumentParser
from sys import stdout, stdin
from warnings import warn

########################################################################


class ArgumentWarning(Warning):
    """Used for warnings related to problematic argument choices."""

    pass


########################################################################


class ValidateQ(Action):
    """Validator for -viewpoint parameter."""

    def __call__(self, parser, args, values, option_string=None):
        """Validates -viewpoint parameter.

        Warns if arguments larger than 100 are passed, reminding that
        they are treated as infinity in the diversity calculation.
        """
        if any([viewpoint > 100 for viewpoint in values]):
            warn(
                "viewpoints > 100.0 defaults to the analytical formula for viewpoint = infinity.",
                category=ArgumentWarning,
            )
        setattr(args, self.dest, values)


########################################################################


def configure_arguments():
    """Creates argument parser.

    Returns
    -------
    argparse.ArgumentParser configured to handle command-line arguments
    for Metacommunity.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        default=stdin,
        type=str,
        help=(
            "A tsv file where the first 3 columns of the file are the"
            " species name, its count, and subcommunity name, and all"
            " following columns are features of that species that"
            " will be used to calculate similarity between species."
        ),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help=(
            "Logging verbosity level. Must be one of DEBUG, INFO,"
            " WARNING, ERROR, CRITICAL (listed in decreasing"
            " verbosity)."
        ),
        default="INFO",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default=stdout,
        type=str,
        help=("A filepath to where the program's output will be saved"),
    )
    parser.add_argument(
        "-s",
        "--similarity_matrix_file",
        type=str,
        help=(
            "The filepath to a tsv file containing a symmetric"
            " similarity matrix. If the file does not exist, one will"
            " be created with the user defined similarity function."
        ),
    )
    parser.add_argument(
        "-t",
        "--store_similarity_matrix",
        help="Do not delete similarity matrix if generated via similarity function.",
    )
    parser.add_argument(
        "-v",
        "--viewpoint",
        nargs="+",
        type=float,
        help="A list of viewpoint parameters.",
        action=ValidateQ,
    )
    return parser
