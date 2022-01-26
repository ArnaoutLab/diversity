"""Module for diversity's argument parser configuration.

Classes
-------
ValidateViewpoint
    Validator for -viewpoint parameter.

Functions
---------
configure_arguments
    Creates argument parser for .
"""
from argparse import Action, ArgumentParser
from sys import stdout, stdin
from warnings import warn

from numpy import inf

from diversity.utilities import ArgumentWarning


class ValidateViewpoint(Action):
    """Validator for -viewpoint parameter."""

    def __call__(self, parser, args, values, option_string=None):
        """Validates -viewpoint parameter.

        Warns if arguments larger than 100 are passed, reminding that
        they are treated as infinity in the diversity calculation.
        """
        if any([viewpoint > 100 and viewpoint != inf for viewpoint in values]):
            warn(
                "viewpoints > 100.0 defaults to the analytical formula"
                " for viewpoint = infinity.",
                category=ArgumentWarning,
            )
        setattr(args, self.dest, values)


def configure_arguments():
    """Creates argument parser.

    Returns
    -------
    argparse.ArgumentParser configured to handle command-line arguments
    for executing diversity as a module.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--count_column",
        help="Header of the column containing the counts.",
        default="count",
    )
    parser.add_argument(
        "-i",
        "--input_filepath",
        help=(
            "A csv or tsv file containing one column for subcommunity"
            " names (with header 'subcommunity', or specified via -u),"
            " one column for species names (with header 'species', or"
            " specified via -p), and one column of counts, which"
            " contains the number of individuals of the row's species"
            " in the row's subcommunity (with header 'count', or"
            " specified via -c)."
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
        "--output_filepath",
        default=stdout,
        help="A filepath to where the program's output will be saved",
    )
    parser.add_argument(
        "-p",
        "--species_column",
        help="Header of the column containing the species names.",
        default="species",
    )
    parser.add_argument(
        "-s",
        "--similarity_matrix_filepath",
        help=(
            "The filepath to a csv or tsv file containing a similarity"
            " for the species in the input file. The file must have a"
            " header listing the species names corresponding to each"
            " column, and column and row ordering must be the same."
        ),
    )
    parser.add_argument(
        "-u",
        "--subcommunity_column",
        help="Header for the column containing the subcommunity names.",
        default="subcommunity",
    )
    parser.add_argument(
        "-v",
        "--viewpoint",
        nargs="+",
        type=float,
        help=(
            "A list of viewpoint parameters. Any non-negative number"
            " (including inf) is valid, but viewpoint parameters"
            " greater than 100 are treated like inf."
        ),
        action=ValidateViewpoint,
    )
    parser.add_argument(
        "-z",
        "--chunk_size",
        type=int,
        help="Number of rows to read at a time from the similarities matrix.",
        default=1,
    )
    return parser
