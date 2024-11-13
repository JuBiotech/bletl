"""
Signal analysis based on simple, deterministic heuristics.
"""
import logging
import typing

import numpy

_log = logging.getLogger(__file__)


def find_do_peak(
    x: numpy.ndarray,
    y: numpy.ndarray,
    *,
    delay_a: float,
    threshold_a: float,
    delay_b: float,
    threshold_b: float,
    initial_delay: float = 1
) -> typing.Optional[int]:
    """Finds the index of a DO peak in the inputs [x] and [y].

    Parameters
    ----------
    x : numpy.ndarray
        Time vector.
    y : numpy.ndarray
        DO vector.
    initial_delay : float
        Hours in the beginning that are not considered.
    delay_a : float
        Hours for which condition A must be fulfilled.
    threshold_a : float
        DO threshold that must be UNDERshot for at least <delay_a> hours.
    delay_b : float
        Hours for which condition B must be fulfilled.
    threshold_b : float
        DO threshold that must be OVERshot for at least <delay_b> hours.

    Returns
    -------
    i_overshoot : int
        Index (w.r.t. inputs [x] and [y]) at which the DO peak was detected.
        `None` is returned if no peak was detected according to the given conditions.
    """
    i_total = len(x)
    i_silencing = numpy.argmax(x > initial_delay)

    i_undershot = None
    for i in range(i_silencing, i_total):
        if y[i] < threshold_a and i_undershot is None:
            # crossing the threshold from above
            i_undershot = i
        elif y[i] > threshold_a:
            # the DO is above the threshold
            i_undershot = None
        if i_undershot is not None:
            undershot_since = x[i] - x[i_undershot]
            if undershot_since >= delay_a:
                # the DO has remained below the threshold for long enough
                break

    i_overshot = None
    if i_undershot is not None:
        for i in range(i_undershot, i_total):
            if y[i] > threshold_b and i_overshot is None:
                # crossing the threshold from below
                i_overshot = i
            elif y[i] < threshold_b:
                # the DO is below the threshold
                i_overshot = None
            if i_overshot is not None:
                overshot_since = x[i] - x[i_overshot]
                if overshot_since >= delay_b:
                    # the DO has remained above the threshold for long enough
                    break

    # Did the series continue long enough after reaching the threshold?
    if i_overshot is not None:
        overshot_since = x[i_total - 1] - x[i_overshot]
        if overshot_since >= delay_b:
            return i_overshot
    return None
