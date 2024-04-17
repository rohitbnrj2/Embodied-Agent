"""
Provides a custom logger object.

Should be used like the following:

```
from cambrian.utils.logger import get_logger

get_logger().fatal("Fatal")
get_logger().error("Error")
get_logger().warn("Warning")
get_logger().info("Information")
get_logger().debug("Debug")
```
"""

import logging


class MjCambrianFileHandler(logging.FileHandler):
    """A file handler which creates the directory if it doesn't exist."""

    def __init__(self, filename, *args, **kwargs):
        # Create the file before calling the super constructor
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        super().__init__(filename, *args, **kwargs)


class MjCambrianTqdmStreamHandler(logging.StreamHandler):
    """A handler that uses tqdm.write to log messages."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            from tqdm import tqdm

            tqdm.write(msg, end=self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class MjCambrianLoggerMaxLevelFilter(logging.Filter):
    """This filter sets a maximum level."""

    def __init__(self, max_level: str):
        self._max_level = logging.getLevelName(max_level)

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= self._max_level


def get_logger(name: str = "cambrian") -> logging.Logger:
    return logging.getLogger(name)
