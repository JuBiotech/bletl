import numpy

__version__ = '0.1.0'


def find_do_peak(x, y, *, delay_a:float, threshold_a:float, delay_b:float, threshold_b:float, initial_delay:float=1):
    """Finds the cycle of the DO peak.
    
    Args:
        x (array): time vector
        y (array): DO vector
        initial_delay (float): hours in the beginning that are not considered
        delay_a (float): hours for which condition A must be fulfilled
        threshold_a (float): DO threshold that must be UNDERshot for at least <delay_a> hours
        delay_b (float): hours for which condition B must be fulfilled
        threshold_b (float): DO threshold that must be OVERshot for at least <delay_b> hours
        
    Returns:
        c_trigger (int): cycle number of the DO peak
    """
    C = len(x)
    c_silencing = numpy.argmax(x > initial_delay)

    c_undershot = None
    for c in range(c_silencing, C):
        if y[c] < threshold_a and c_undershot is None:
            # crossing the threshold from above
            c_undershot = c
        elif y[c] > threshold_a:
            # the DO is above the threshold
            c_undershot = None
        if c_undershot is not None:
            undershot_since = x[c] - x[c_undershot]
            if undershot_since > delay_a:
                # the DO has remained below the threshold for long enough
                break

    c_overshot = None
    if c_undershot is not None:
        for c in range(c_undershot, C):
            if y[c] > threshold_b and c_overshot is None:
                # crossing the threshold from below
                c_overshot = c
            elif y[c] < threshold_b:
                # the DO is below the threshold
                c_overshot = None
            if c_overshot is not None:
                overshot_since = x[c] - x[c_overshot]
                if overshot_since > delay_b:
                    # the DO has remained above the threshold for long enough
                    break
    return c_overshot
