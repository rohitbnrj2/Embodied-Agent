"""
Provides a custom logger object.

Should be used like the following:

```
from cambrian.utils.logger import LOGGER

LOGGER.fatal("Fatal")
LOGGER.error("Error")
LOGGER.warn("Warning")
LOGGER.info("Information")
LOGGER.debug("Debug")
```
"""

import logging
import logging.config
from typing import Optional

from cambrian.utils.config import MjCambrianConfig


class MjCambrianTqdmStreamHandler(logging.StreamHandler):
    """A handler that uses tqdm.write to log messages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, max_level: int = logging.INFO):
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= self.max_level


def get_logger(config: Optional[MjCambrianConfig] = None, *, name: str = "cambrian") -> logging.Logger:
    """Get/configure the logger."""
    if config is None:
        assert name in logging.root.manager.loggerDict, f"Logger {name} does not exist"
        return logging.getLogger(name)

    assert name not in logging.root.manager.loggerDict, f"Logger {name} already exists"

    for handler_name, handler in config.select("logging_config.handlers").items():
        if filename := handler.get("filename"):
            import pathlib

            if not pathlib.Path(filename).parent.exists():
                del config.logging_config["handlers"][handler_name]
                for logger in config.logging_config["loggers"].values():
                    logger["handlers"].remove(handler_name)

    logging.config.dictConfig(config.logging_config)

    return logging.getLogger(name)