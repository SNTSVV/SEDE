"""
This module contains the infrastructure to setup a convenient logging infrastructure.

TODO: It is very simple but I keep copying it in different projects such that it might be
useful create a stand-alone python package from it that can be included into other projects.
"""

import logging
import datetime

from pathlib import Path


def _log_level_from_string(level: str):
    """
    Extract the logging level from a string.
    Everything will be converted to lowercase, so capitalization does not matter.

    Arguments
    ---------
    level: str
        The string-form of the logging level.

    Returns
    -------
    A level that can be parsed by the `logging` module.

    Raises
    ------
    ValueError:
        If `level` is neither of debug, info, warning, error, critical.

    Examples
    --------
    >>> import logging
    >>> _log_level_from_string('info') == logging.INFO
    True
    """
    lower = level.lower()
    if lower == "debug":
        return logging.DEBUG
    elif lower == "info":
        return logging.INFO
    elif lower == "warning":
        return logging.WARNING
    elif lower == "error":
        return logging.ERROR
    elif lower == "critical":
        return logging.CRITICAL
    else:
        raise ValueError(f"'{level}' is not a valid logging-level indicator.")


def _log_level_to_string(log_level: int):
    """
    Convert the log level from an integer to a string.

    Arguments
    ---------
    log_level: int
        The log-level (10 - debug, 20 - info, 30 - warning, 40 - error, 50 - critical)

    Raises
    ------
    ValueError
        If the received `log_level` value is not valid.

    Example
    -------
    >>> _log_level_to_string(30)
    'WARNING'
    """
    if log_level == 10:
        return "DEBUG"
    elif log_level == 20:
        return "INFO"
    elif log_level == 30:
        return "WARNING"
    elif log_level == 40:
        return "ERROR"
    elif log_level == 50:
        return "CRITICAL"
    else:
        raise ValueError(f"'{log_level}' is not a valid log-level specification")


def setup_logger(
    logfile_path: Path = None,
    log_level: int = logging.INFO,
    disable_existing=True,
    formatter="%(asctime)s - [%(levelname)-7s]: %(message)s",
    mode="w",
    add_timestep_top=False,
    datefmt="%Y-%m-%d,%H:%M:%S",
):
    """
    Setup the logger.

    Arguments
    ---------
    logfile_path: Path
        The path to the file where the log shall be written to.
        If logging shall only be performed to the standard output, set `logfile_path=None` which is the default.
    log_level: int
        The minimal importance level a log message needs to have in order to be displayed.
        Defaults to `logging.INFO`.
    disable_existing: bool
        Whether or not to disable existing loggers. Defaults to `True`.
    formatter: str
        Specify the format of the logging message (e.g. log level, log message, date, time and their arrangments).
    mode: str
        Specify whether to append (`a`) to the log file or erase the previous content (`w`). The default is `w`.
    add_timestep_top: bool
        Whether or not to add the date and time to the top of the log file.
        This might be useful if `formatter` does not contain date and time information and if `mode=a` in order to
        distinguish when a new logging process starts.
    datefmt: str
        Formatter for the date and time strings that can appear in the log messages.

    """
    # Disable existing loggers if requested
    if disable_existing:
        import logging.config

        logging.config.dictConfig(
            {"version": 1, "disable_existing_loggers": True, }
        )

    # Start by specifying the logging format and the logging level
    logging.basicConfig(level=log_level, format=formatter, datefmt=datefmt)

    # Also log to a file?
    if logfile_path is not None:
        root = logging.getLogger()
        handler = logging.FileHandler(logfile_path, mode="w")
        handler.setLevel(log_level)
        formatter = logging.Formatter(formatter, datefmt=datefmt)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if add_timestep_top:
        logging.info(
            f"New logger set up on: {datetime.datetime.now().strftime(datefmt)}"
        )
    logging.debug(f"Using log level: {_log_level_to_string(log_level)}")
    if logfile_path:
        logging.debug(f"Logging to file: {logfile_path.resolve()}")


def log_and_raise(message: str, error_type: Exception):
    """
    Log a message while simultaneously raising an exception of type `error_type` with the same message.

    Parameters
    ----------
    message: str
        The message for the log entry and the exception description.
    error_type: type
        The type of the exception to raise

    Raises
    ------
    error_type
        The exception to raise.
    """
    logging.critical(message)
    raise error_type(message)
