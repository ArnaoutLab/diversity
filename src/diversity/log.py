"""Helper module for logging operations.

Constants
---------
LOG_HANDLER: logging.StreamHandler
    Handler used for logging.
LOGGER: logging.Logger
    Multiprocessing-safe logger.
"""
from logging import (
    Formatter,
    StreamHandler,
)
from multiprocessing import get_logger

date_format = "%Y-%m-%dT%H:%M:%S%z"
logging_format = (
    "%(asctime)s\t(%(processName)s, %(threadName)s)\t%(levelname)s\t%(message)s"
)

LOGGER = get_logger()
formatter = Formatter(fmt=logging_format, datefmt=date_format)
LOG_HANDLER = StreamHandler()
LOG_HANDLER.setFormatter(formatter)
LOGGER.addHandler(LOG_HANDLER)
