import abc
import inspect
import logging
import time
import typing

import fastprogress
import numpy
import pandas
import tsfresh

from . import splines
from .types import BLData, FilterTimeSeries

_log = logging.getLogger(__file__)


class Extractor(abc.ABC):
    """Common base class for all feature extractors."""

    def get_methods(self) -> typing.Dict[str, typing.Callable[[numpy.ndarray, numpy.ndarray], float]]:
        """Returns the extration methods by name.

        All classmethods that are named `extract_*` are considered feature
        extration methods.

        Returns
        -------
        methods : dict
            dictionary of { name : callable }
        """
        methods = {}
        for name, method in inspect.getmembers(self):
            if name.startswith("extract_"):
                methods[name[7:]] = method
        return methods


class StatisticalFeatureExtractor(Extractor):
    """Class for statistical feature extraction."""

    def extract_mean(self, x, y):
        """Extracts the mean of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            mean
        """
        return numpy.mean(y)

    def extract_min(self, x, y):
        """Extracts the minimum of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            minimum
        """
        return numpy.min(y)

    def extract_time_min(self, x, y):
        """Extracts the time at the minimum of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            time at the minimum
        """
        return x[numpy.argmin(y)]

    def extract_max(self, x, y):
        """Extracts the maximum of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            maximum
        """
        return numpy.max(y)

    def extract_time_max(self, x, y):
        """Extracts the time at the maximum of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            time at the maximum
        """
        return x[numpy.argmax(y)]

    def extract_span(self, x, y):
        """Extracts the span between minimum and maximum.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            span
        """
        return numpy.max(y) - numpy.min(y)

    def extract_median(self, x, y):
        """Extracts the median of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            median
        """
        return numpy.median(y)

    def extract_stan_dev(self, x, y):
        """Extracts the standard deviation of y.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            standard deviation
        """
        return numpy.std(y)


class DOFeatureExtractor(Extractor):
    """Class for feature extraction of dissolved oxygen"""

    def extract_peak(self, x, y):
        """Extracts the duration of DO < 5

        Returns
        -------
        result : float
            duration of DO < 5
        """
        tmp = 0
        boolean = False
        time = 0
        # if values are under 5, the time spans are calculated and added together
        for i in range(len(y)):
            if y[i] < 5 and boolean is False:
                boolean = True
                tmp = i
            if y[i] > 5 and boolean is True:
                boolean = False
                time = time + x[i] - x[tmp]
        if boolean is True:
            time = time + x[len(y) - 1] - x[tmp]
        return time


class BSFeatureExtractor(Extractor):
    """Class for feature extraction of backscatter."""

    def extract_inflection_point_t(self, x, y):
        """Extracts value at the turning point.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            turning point
        """
        spline = splines.get_crossvalidated_spline(x, y)
        y_der1 = spline.derivative(1)(x)
        return y[numpy.argmax(y_der1)]

    def extract_inflection_point_y(self, x, y):
        """Extracts time at the turning point.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            time of turning point
        """
        spline = splines.get_crossvalidated_spline(x, y)
        y_der1 = spline.derivative(1)(x)
        return x[numpy.argmax(y_der1)]

    def extract_mue_median(self, x, y):
        """Extracts median of mue at the exponential phase.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            median
        """
        # calculate value, which lies in exponential phase
        mid_y = (numpy.max(y) - numpy.min(y)) / 2
        # get index of a value, which lies in exponential phase
        mid_i = int(numpy.where(y > mid_y)[0][0] / 2)
        # get mue
        bsdata = FilterTimeSeries(pandas.DataFrame(x), pandas.DataFrame(y))
        mue = splines.get_mue(bsdata, method="us")
        var = 0.1  # allowed variance
        mue_val = numpy.array(mue.value)
        list_values = [mue_val[mid_i]]  # list of mue values, which lie in ex. phase
        l_i = mid_i - 1
        r_i = mid_i + 1
        # checks if mue_val[l_i] lies in variance of mue_val from mid_i, goes left
        while l_i >= 0 and mue_val[l_i] < mue_val[mid_i] + var and mue_val[l_i] > mue_val[mid_i] - var:
            list_values.append(mue_val[l_i])
            l_i = l_i - 1
        # checks if mue_val[l_i] lies in variance of mue_val from mid_i, goes right
        while (
            r_i < len(mue_val) and mue_val[r_i] < mue_val[mid_i] + var and mue_val[r_i] > mue_val[mid_i] - var
        ):
            list_values.append(mue_val[r_i])
            r_i = r_i + 1
        return numpy.median(list_values)


class pHFeatureExtractor(Extractor):
    """Class for feature extraction of pH."""

    def extract_sum_of_reduction(self, x, y):
        """Extracts sum of reduction.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            sum of reduction
        """
        spline = splines.get_crossvalidated_spline(x, y)
        y = spline(x)
        changes = numpy.diff(y)
        return numpy.sum(changes[changes < 0])

    def extract_sum_of_increase(self, x, y):
        """Extracts sum of increase.

        Parameters
        ----------
        x : numpy.ndarray
            list of time values
        y : numpy.ndarray
            list of values

        Returns
        -------
        result : float
            sum of increase
        """
        spline = splines.get_crossvalidated_spline(x, y)
        y = spline(x)
        changes = numpy.diff(y)
        return numpy.sum(changes[changes > 0])


class TSFreshExtractor:
    """Class for feature extraction with tsfresh."""

    def __init__(self, **kwargs):
        """Creates a TSFreshExtractor object.

        Parameters
        ----------
        **kwargs
            Keyword arguments to forward to `tsfresh.extract_features`.
            For example `TSFreshExtractor(n_jobs=0)` to disable multiprocessing.
        """
        self._kwargs = kwargs
        super().__init__()

    def _extract(self, data):
        """Extracts data with tsfresh.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame of a filterset

        Returns
        -------
        result : pandas.DataFrame
            DataFrame with extracted features
        """
        kwargs = dict(column_id="id", column_sort="time")
        kwargs.update(self._kwargs)
        return tsfresh.extract_features(data, **kwargs)


def from_bldata(
    bldata: BLData,
    extractors: typing.Dict[str, typing.Sequence[Extractor]],
    last_cycles: typing.Optional[typing.Dict[str, int]] = None,
    *,
    take_wells: typing.Optional[typing.Iterable[str]] = None,
) -> pandas.DataFrame:
    """Apply feature extractors to a dataset.

    Parameters
    ----------
    data : BLData
        a dataset to extract from
    extractors : dict
        map of { filterset : [extractor, ...] }
    last_cycles :  optional, dict
        maps well ids to the number of the last cycle to consider
    take_wells : iterable
        List or set of wells that should be extracted from.
        This should be used to remove wells that are sampled
        too early to have meaningful time series features.

    Returns
    -------
    result : pandas.DataFrame
        well-indexed features
    """
    # create useful local variables from inputs
    if not last_cycles:
        last_cycles = {}
    filtersets = set(extractors.keys())
    for fs in filtersets.difference(set(bldata.keys())):
        _log.warning('No "%s" filterset in the data. Skipping extractors.', fs)

    # Identify wells of interest
    wells = take_wells or bldata[list(filtersets)[0]].value.columns
    wells = tuple(sorted(wells))

    extraction_methods = {
        f"{fs}_{mname}": method
        for fs, fs_extractors in extractors.items()
        for extractor in fs_extractors
        if not isinstance(extractor, TSFreshExtractor)
        for mname, method in extractor.get_methods().items()
    }
    ts_extractors = {
        fs: extractor
        for fs, fs_extractors in extractors.items()
        for extractor in fs_extractors
        if isinstance(extractor, TSFreshExtractor)
    }

    _log.info("Applying custom extractors to %i wells.", len(wells))
    s_time = time.time()
    df_result = pandas.DataFrame(index=wells)
    for well in fastprogress.progress_bar(wells):
        for mname, method in extraction_methods.items():
            t, y = bldata[mname.split("__")[0]].get_timeseries(well, last_cycle=last_cycles.get(well))
            df_result.loc[well, mname] = method(t, y)
    narrow = bldata.get_unified_narrow_data(last_cycles=last_cycles)
    narrow = narrow[narrow.well.isin(wells)]
    for fs, ts_extractor in ts_extractors.items():
        data = pandas.DataFrame(columns=["id", "time", "x"])
        data["id"] = narrow["well"]
        data["time"] = narrow["time"]
        data["x"] = narrow[fs]
        _log.info("Applying tsfresh extractor to %s\n", fs)
        df_fs = ts_extractor._extract(data)
        # Rename columns to include the filterset name
        df_fs.columns = [f"{fs}_{col}" for col in df_fs.columns]
        # Add columns with new features for this filterset to the result
        df_fs.index = [i for i in df_fs.index]
        df_result = pandas.concat([df_result, df_fs], axis=1)
    _log.info("Extraction finished in: %s minutes", round((time.time() - s_time) / 60, ndigits=1))
    return df_result
