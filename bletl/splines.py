import copy
import logging
import multiprocessing
import numbers
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import csaps
import joblib
import numpy
import pandas
import scipy.interpolate
import scipy.optimize
import scipy.stats

from .types import FilterTimeSeries

logger = logging.getLogger(__file__)


class UnivariateCubicSmoothingSpline(csaps.CubicSmoothingSpline):
    """Overrides `csaps` type to align its API with the `scipy.interpolate` splines."""

    def derivative(self, order: int = 1, *, epsilon=0.001) -> Callable[[numpy.ndarray], numpy.ndarray]:
        """Returns the derivative of the spline.

        Parameters
        ----------
        order : int
            Order of derivative.
        epsilon : float
            Epsilon to calculate derivative by difference quotient.

        Returns
        -------
        derivative : callable
            A function that can be called with a numpy array of coordinates at which to apply
            the difference quotient derivative calculation.
        """
        if order == 1:
            return lambda x: (self(x + epsilon) - self(x - epsilon)) / (2 * epsilon)
        elif order > 1:
            der = self.derivative(order - 1)
            der_f = lambda x: der(x)
            return lambda x: (der_f(x + epsilon) - der_f(x - epsilon)) / (2 * epsilon)
        raise NotImplementedError(
            f"{order}-order derivatives are not implemented for the UnivariateCubicSmoothingSpline"
        )

    def __call__(
        self,
        x: csaps.UnivariateDataType,
        nu: Optional[int] = None,
        extrapolate: Optional[Union[bool, str]] = None,
    ) -> numpy.ndarray:
        """Evaluate the spline at some coordinates.

        This method overrides the implementation of the base type
        to align it with the API of the `scipy.interpolate` splines.

        Parameters
        ----------
        x : float, numpy.ndarray
            One or more coordinates at which the spline should be evaluated.
        """
        # scipy splines can be called on scalars
        xi_arr = numpy.atleast_1d(numpy.asarray(x))
        result = super().__call__(xi_arr)
        if isinstance(x, (int, float)):
            return result[0]
        return result


def _normalize_smoothing_factor(method: str, smooth: float, x: numpy.ndarray, y: numpy.ndarray) -> float:
    """Normalization of smoothing factor from the interval [0,1] to the value required by the Spline type.

    Parameters
    ----------
    method : str
        Kind of spline.
    smooth : float
        Smoothing factor in the interval [0,1], where 0 results in interpolation through all data points.
    x : numpy.ndarray
        Timepoints of the entire dataset.
    y : numpy.ndarray
        Values of the entire dataset.

    Returns
    -------
    smoothing_factor : float
        Smoothing factor transformed to the domain used by the spline `method`.
    """
    if method == "ucss":
        return 1 - smooth
    elif method == "us":
        # the s parameter of the UnivariateSpline describes the maximum sum squared error of the fit
        # a reasonable upper bound for this SSE is the SSE of a linear fit
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        max_sse = numpy.sum(numpy.square((intercept + slope * x) - y))
        return smooth * max_sse
    else:
        raise NotImplementedError(f'Unknown method "{method}"')


def _evaluate_smoothing_factor(smoothing_factor: float, timepoints, values, k: int, method: str) -> float:
    """Computes mean sum of squared residuals over K folds of the dataset.

    Parameters
    ----------
    smoothing_factor : float
        Smoothing factor for the univariate spline.
    timepoints : numpy.ndarray
        Timepoints of the data.
    values : numpy.ndarray
        Values of the data.
    k : int
        Number of train/test splits.
    method : str
        Kind of spline.
        Options: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline

    Returns
    -------
    mssr : float
        Mean sum of squared residuals of the splines fitted to data subsets.
    """
    smoothing_factor_arr = numpy.atleast_1d(smoothing_factor)
    if k < 2:
        raise ValueError(f"Need k≥2 splits for crossvalidation. Setting was k={k}.")
    if len(values) < 3 * k:
        raise ValueError(
            f"Time series of {len(values)} elements is too short. " f"Need at least a length of 3*k ({3*k})."
        )
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
                smoothing_factor=smoothing_factor_arr,
                method=method,
            )
        )
    return float(numpy.mean(ssrs))


def _evaluate_spline_test_error(
    x: numpy.ndarray,
    y: numpy.ndarray,
    train_idxs: numpy.ndarray,
    test_idxs: numpy.ndarray,
    smoothing_factor: numpy.ndarray,
    method: str,
) -> float:
    """Fits spline to a test set and returns the sum of squared error on the test set.

    Parameters
    ----------
    x : numpy.ndarray
        Timepoints
    y : numpy.ndarray
        Values
    train_idxs : numpy.ndarray
        Indices or bool mask of training data points.
    test_idxs : numpy.ndarray
        Indices or bool mask of test data points.
    smoothing_factor : float
        Smoothing factor for the univariate spline.
    method : str
        Kind of spline.
        Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline

    Returns
    -------
    ssr : float
        Sum of squared residuals on test set.
    """
    smooth = _normalize_smoothing_factor(method, smoothing_factor[0], x, y)
    if method == "ucss":
        spline = csaps.CubicSmoothingSpline(x[train_idxs], y[train_idxs], smooth=smooth)
    elif method == "us":
        spline = scipy.interpolate.UnivariateSpline(x[train_idxs], y[train_idxs], s=smooth)
    else:
        raise NotImplementedError(f'Unknown method "{method}"')
    y_val_pred = spline(x[test_idxs])
    return numpy.sum(numpy.square(y_val_pred - y[test_idxs]))


def get_crossvalidated_spline(
    x: numpy.ndarray,
    y: numpy.ndarray,
    k_folds: int = 5,
    method: str = "us",
    bounds: Tuple[float, float] = (0.001, 1),
) -> Union[UnivariateCubicSmoothingSpline, scipy.interpolate.UnivariateSpline]:
    """Returns spline with k-fold crossvalidated smoothing factor

    Parameters
    ----------
    x : numpy.ndarray
        Time vector
    y : numpy.ndarray
        Value vector
    k_folds : int
        Number of splits used for cross-validation.
    method : str
        Kind of spline.
            Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline
    bounds : tuple
        Lower and upper bound for the smoothing factor. Must not exceed the [0, 1] interval.

    Returns
    -------
    spline : UnivariateCubicSmoothingSpline or scipy.interpolate.UnivariateSpline
        Spline of the specified `method` fitted to the data.
        Its optimal smoothing factor is determined by k-fold crossvalidation.
    """
    opt = scipy.optimize.differential_evolution(
        _evaluate_smoothing_factor, bounds=[bounds], args=(x, y, k_folds, method)
    )
    smooth = _normalize_smoothing_factor(method, opt.x[0], x, y)
    if method == "ucss":
        return UnivariateCubicSmoothingSpline(x, y, smooth=smooth)
    elif method == "us":
        return scipy.interpolate.UnivariateSpline(x, y, s=smooth)
    else:
        raise NotImplementedError(f'Unknown method "{method}"')


def get_multiple_splines(
    fts: FilterTimeSeries,
    wells: Sequence[str],
    k_folds: int = 5,
    method: str = "us",
    last_cycle: Optional[int] = None,
) -> List[Tuple[str, Union[UnivariateCubicSmoothingSpline, scipy.interpolate.UnivariateSpline]]]:
    """Returns multiple splines with k-fold crossvalidated smoothing factor

    Parameters
    ----------
    fts : FilterTimeSeries
        FilterTimeSeries containing timepoints and observed data-
    wells : list of str
        Wells to fit splines to.
    k_folds : int
        "k"s for cross-validation
    method : str
        Kind of spline.
        Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline

    Returns
    -------
    splines : dict
        Dict with {well:spline} for each well of "wells"
    """
    x, y = fts.get_timeseries(wells[0])
    if last_cycle is None:
        last_cycle = len(x)
    elif last_cycle > len(x):
        raise ValueError("Please change last_cycle.")

    def get_spline_parallel(arg):
        well, timepoints, values, k_folds = arg
        spline = get_crossvalidated_spline(timepoints, values, k_folds, method=method)
        return (well, spline)

    args_get_spline = []
    for well in wells:
        timepoints, values = fts.get_timeseries(well)
        args = (
            copy.deepcopy(well),
            copy.deepcopy(timepoints[:last_cycle]),
            copy.deepcopy(values[:last_cycle]),
            copy.deepcopy(k_folds),
        )
        args_get_spline.append(args)
    if len(wells) > 1:
        try:
            return joblib.Parallel(n_jobs=multiprocessing.cpu_count(), verbose=11)(
                map(joblib.delayed(get_spline_parallel), args_get_spline)
            )
        except pickle.PicklingError:
            logger.warning("Parallelization failed. Retrying without parallelization.")
    return list(map(get_spline_parallel, args_get_spline))


def get_mue(
    bsdata: FilterTimeSeries,
    wells: Union[Sequence[str], str] = "all",
    blank: Union[float, str, Dict[str, float]] = "first",
    k_folds: int = 5,
    method: str = "us",
    last_cycle: Optional[int] = None,
):
    """Approximation of specific growth rate over time via spline approximation using splines with k-fold cross validated smoothing factor

    Parameters
    ----------
    bsdata : FilterTimeSeries
        FilterTimeSeries containing timepoints and backscatter data.
    wells : 'all' or list
        List of wells to calculate specific growth rate for. 'all' calculates for all wells.
    blank : 'first', float or dict
        Blanks to use for specific growth rate calculation.
        Options:
        - 'first': Use first data point.
        - float: Apply this blank value to all wells.
        - dict: Containing well id as key and scalar or vector as blank value(s) for the respective well.
    k_folds : int
        "k"s for cross-validation.
    method : str
        Kind of Spline.
        Choices: "ucss" UnivariateCubicSmoothingSpline, "us" UnivariateSpline
    last_cycle : int, optional
        Ignores data after last_cycle

    Returns
    -------
    filtertimeseries : FilterTimeSeries
        FilterTimeSeries with specific growth rate over time.

    Raises
    ------
    ValueError
        If blanks are not provided for every well,
        of if an invalid blank option is passed.
    """
    # check inputs
    if wells == "all":
        wells = list(bsdata.time.columns)
    if blank == "first":
        blank_dict = {well: data.iloc[0] for well, data in bsdata.value.items()}
    elif isinstance(blank, numbers.Number):
        blank_dict = {well: blank for well, data in bsdata.value.items()}
    elif isinstance(blank, dict):
        if set(blank.keys()) != set(wells):
            raise ValueError("Please provide blanks for every well")
        blank_dict = blank
    else:
        raise ValueError("Please provide proper blank option.")
    # run spline fitting
    results = get_multiple_splines(bsdata, wells, method=method, last_cycle=last_cycle)
    # compute derivatives
    time = {}
    mues = {}
    for result in results:
        well = result[0]
        spline = result[1]
        der = spline.derivative(1)
        if blank == "first":
            time[well] = bsdata.time[well][1:last_cycle]
        else:
            time[well] = bsdata.time[well][:last_cycle]
        mues[well] = der(time[well]) / (spline(time[well]) - blank_dict[well])
    # summarize into FilterTimeSeries
    filtertimeseries = FilterTimeSeries(
        pandas.DataFrame(time),
        pandas.DataFrame(mues),
    )
    return filtertimeseries
