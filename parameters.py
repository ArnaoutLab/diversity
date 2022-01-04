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
from warnings import warn

########################################################################


class ArgumentWarning(Warning):
    """Used for warnings related to problematic argument choices."""
    pass

########################################################################


class ValidateQ(Action):
    """Validator for -q parameter."""

    def __call__(self, parser, args, values, option_string=None):
        """Validates -q parameter.

        Warns if arguments larger than 100 are passed, reminding that
        they are treated as infinity in the diversity calculation.
        """
        if any([q > 100 for q in values]):
            warn("q > 100.0 defaults to the analytical formula for q = inf.",
                 category=ArgumentWarning)
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
        "filepath",
        type=str,
        help=("A csv file where the first 3 columns of the file are the"
              " species name, its count, and subcommunity name, and all"
              " following columns are features of that species that"
              " will be used to calculate similarity between species."))
    parser.add_argument(
        "-l", "--log_level",
        help=("Logging verbosity level. Must be one of DEBUG, INFO,"
              " WARNING, ERROR, CRITICAL (listed in decreasing"
              " verbosity)."),
        default="INFO")
    parser.add_argument(
        "-q",
        nargs='+',
        type=float,
        help="A list of viewpoint parameters.",
        action=ValidateQ)
    parser.add_argument(
        "Z",
        type=str,
        help=("The filepath to a csv file containing a symmetric"
              " similarity matrix. If the file does not exist, one will"
              " be created with the user defined similarity function."))
    return parser
