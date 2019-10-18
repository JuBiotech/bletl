import copy
import joblib
import multiprocessing
import numbers
import numpy
import pandas
import scipy.optimize
import scipy.interpolate
import csaps
import bletl

__version__ = '1.0.0'

class UnivariateCubicSmoothingSpline(csaps.UnivariateCubicSmoothingSpline):
    """add a function to UnivarianteCubicSmoothingSpline
    """
    def derivative(self, order:int=1, *, epsilon=0.001):
        """returns derivative of UnivarianteCubicSmoothingSpline
        Args:
            order(int): order of derivative
            epsilon(float): epsilon to calculate derivative
        Returns:
            derivative
        """
        if order == 1:
            return lambda x: (self(x + epsilon) - self(x - epsilon)) / (2 * epsilon)
        elif order == 2:
            der = self.derivative(1)
            der1 = lambda x: der(x)
            return lambda x: (der1(x + epsilon) - der1(x - epsilon)) / (2 * epsilon)
        raise NotImplementedError(f'{order}-order derivatives are not implemented for the UnivariateCubicSmoothingSpline')


def _normalize_smoothing_factor(method:str, smooth:float, y):
    """normalization of smoothing factor
    Args:
        method(str): kind of spline 
        smooth(float): smoothing factor
        y: spline
    Returns:
        normailzed smoothing factor
    """
    if method == 'ucss':
        return 1 - smooth
    elif method == 'us':
        amplitude = (numpy.max(y) - numpy.min(y)) / 2
        return smooth * amplitude * 10
    else:
        raise NotImplementedError(f'Unknown method "{method}"')


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


def _evaluate_smoothing_factor(smoothing_factor:float, timepoints, values, k:int, method:str) -> float:
    """Computes mean sum of squared residuals over K folds of the dataset.
    
    Args:
        smoothing_factor (float): smoothing factor for the univariate spline
        timepoints (array): timepoints of the data
        values (array): values of the data
        k (int): number of train/test splits
        method (str): Kind of spline, Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline
        
    Returns:
        mssr (float): mean sum of squared residuals
    """
    ssrs = []
    for kshift in range(k):
        train_mask = numpy.ones_like(timepoints, dtype=bool)
        # drop every K-th element from the trainig set, beginning at kshift
        train_mask[kshift::k] = False
        test_mask = numpy.invert(train_mask)
        ssrs.append(
            _evaluate_spline_test_error(
                x=timepoints,
                y=values,
                train_idxs=train_mask,
                test_idxs=test_mask,
                smoothing_factor=smoothing_factor,
                method=method
            )
        )
    return numpy.mean(ssrs)


def _evaluate_spline_test_error(x, y, train_idxs, test_idxs, smoothing_factor:float, method:str) -> float:
    """Fits spline to a test set and returns the sum of squared error on the test set.

    Args:
        x (array): timepoints
        y (array): values
        train_idxs (array): indices or bool mask of training data points
        test_idxs (array): indices or bool mask of test data points
        smoothing_factor (float): smoothing factor for the univariate spline
        method (str): Kind of spline, Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline

    Returns:
        ssr (float): sum of squared residuals on test set
    """
    smooth = _normalize_smoothing_factor(method, smoothing_factor[0], y)
    if method == 'ucss':
        spline = csaps.UnivariateCubicSmoothingSpline(x[train_idxs], y[train_idxs], smooth=smooth)
    elif method == 'us':
        spline = scipy.interpolate.UnivariateSpline(x[train_idxs], y[train_idxs], s=smooth)
    else:
        raise NotImplementedError(f'Unknown method "{method}"')
    y_val_pred = spline(x[test_idxs])
    return numpy.sum(numpy.square(y_val_pred - y[test_idxs]))      


def _crossvalidate_smoothing_spline(x, y, k_folds:int=5, method:str='ucss', bounds=(0,1)):
    """Returns spline with k-fold crossvalidated smoothing factor
    
    Args:
        x (array): time vector
        y (array): value vector
        k_folds (int): "k"s for cross-validation
        method (str): Kind of spline, Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline  
    Returns:
        spline (scipy.interpolate.UnivariateSpline): Spline with k-fold crossvalidated smoothing factor
    """
    opt = scipy.optimize.differential_evolution(_evaluate_smoothing_factor,
        bounds=[bounds],
        args=(x, y, k_folds, method)
    )
    smooth = _normalize_smoothing_factor(method, opt.x[0], y)
    if method == 'ucss':
        return UnivariateCubicSmoothingSpline(x, y, smooth=smooth)
    elif method == 'us':
        return scipy.interpolate.UnivariateSpline(x, y, s=smooth)
    else:
        raise NotImplementedError(f'Unknown method "{method}"')


def _get_multiple_splines(bsdata:bletl.core.FilterTimeSeries, wells:list, k_folds:int=5, method:str='ucss'):
    """Returns multiple splines with k-fold crossvalidated smoothing factor
    
    Args:
        bsdata (bletl.core.FilterTimeSeries): FilterTimeSeries containing timepoints and backscatter data
        wells (list): List of wells to calculate specific growth rate for.
        k_folds (int): "k"s for cross-validation
        method (str): Kind of spline, Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline 
    Returns:
        splines (dict): Dict with well:spline for each well of "wells"
    """
    def get_spline_parallel(arg):
        well, timepoints, values, k_folds = arg
        spline = _crossvalidate_smoothing_spline(timepoints, values, k_folds, method=method)
        return (well, spline)

    args_get_spline = []
    for well in wells:
        timepoints, values = bsdata.get_timeseries(well)
        args = (copy.deepcopy(well), copy.deepcopy(timepoints), copy.deepcopy(values), copy.deepcopy(k_folds))
        args_get_spline.append(args)

    return joblib.Parallel(n_jobs=multiprocessing.cpu_count(), verbose=11)(map(joblib.delayed(get_spline_parallel), args_get_spline))


def get_mue(bsdata:bletl.core.FilterTimeSeries, wells='all', blank='first', k_folds:int=5, method:str='ucss'):
    """Approximation of specific growth rate over time via spline approximation using splines with k-fold cross validated smoothing factor
    
    Args:
        bsdata (bletl.core.FilterTimeSeries): FilterTimeSeries containing timepoints and backscatter data
        wells ('all' or list): List of wells to calculate specific growth rate for. 'all' calculates for all wells.
        blanks ('first', float or dict): Blanks to use for specific growth rate calculation. Options:
            - 'first' (str): Use first data point 
            - (float): Apply this blank value to all wells
            - (dict): Containing well id as key and scalar or vector as blank value(s) for the respective well
        k_folds (int): "k"s for cross-validation
        method (str): Calculates Spline, Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline
    Returns:
        filtertimeseries (bletl.core.FilterTimeSeries): FilterTimeSeries with specific growth rate over time

    Raises:
        ValueError: if blanks are not provided for every well
        ValueError: invalid blank option
    """
    # check inputs
    if wells == 'all':
        wells = list(bsdata.time.columns)

    if blank == 'first':
        blank_dict = {well:data.iloc[0] for well, data in bsdata.value.iteritems()}
    elif isinstance(blank, numbers.Number):
        blank_dict = {well:blank for well, data in bsdata.value.iteritems()}
    elif isinstance(blank, dict):
        if set(blank.keys()) != set(wells):
            raise ValueError('Please provide blanks for every well')
        blank_dict = blank
    else:
        raise ValueError('Please provide proper blank option.')

    # run spline fitting
    results = _get_multiple_splines(bsdata, wells, method=method)
    
    # compute derivatives
    time = {}
    mues = {}
    for result in results:
        well = result[0]
        spline = result[1]
        der = spline.derivative(1)
        if blank == 'first':
            time[well] = bsdata.time[well][1:]
        else:
            time[well] = bsdata.time[well]

        mues[well] = der(time[well])/(spline(time[well]) - blank_dict[well])
    # summarize into FilterTimeSeries
    filtertimeseries = bletl.core.FilterTimeSeries(
        pandas.DataFrame(time),
        pandas.DataFrame(mues),
    )
    return filtertimeseries
