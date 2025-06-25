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
import functools
import time

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
LOGGING_FORMAT = (
    "%(asctime)s\t(%(processName)s, %(threadName)s)\t%(levelname)s\t%(message)s"
)

LOGGER = get_logger()
formatter = Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT)
LOG_HANDLER = StreamHandler()
LOG_HANDLER.setFormatter(formatter)
LOGGER.addHandler(LOG_HANDLER)


def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        ms = int((t1 - t0) * 1000)
        print(f"Time for {func.__qualname__}: {ms}")
        return result

    return wrapper
