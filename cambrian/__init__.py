"""Cambrian package init file."""

import signal

import cambrian.agents  # noqa
import cambrian.envs  # noqa
import cambrian.eyes  # noqa

__author__ = "Camera Culture (a2cc@media.mit.edu)"
"""Camera Culture (a2cc@media.mit.edu)"""
__license__ = "BSD3"
"""BSD3"""


def _signal_handler(sig, frame):
    """Signal handler that will exit if ctrl+c is recorded in the terminal window.

    Allows easier exiting of a matplotlib plot

    Args:
        sig (int): Signal number
        frame (int): ?
    """

    import sys

    sys.exit(0)


# setup the signal listener to listen for the interrupt signal (ctrl+c)
signal.signal(signal.SIGINT, _signal_handler)

del signal
