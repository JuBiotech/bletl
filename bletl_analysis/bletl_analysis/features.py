import abc
import inspect
import logging
import numpy
import pandas
import time
import typing
import tsfresh

import bletl
import bletl_analysis


_log = logging.getLogger(__file__)


class Extractor(abc.ABC):
    """ Common base class for all feature extractors. """  

    def get_methods(self) -> typing.Dict[str, typing.Callable[[numpy.ndarray, numpy.ndarray], float]]:
        """ Returns the extration methods by name.

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
    """ Class for statistical feature extraction."""
    
    def extract_mean(self, x, y):
        """ Extracts the mean of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            mean
        """
        return numpy.mean(y)
    
    def extract_min(self, x, y):
        """ Extracts the minimum of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            minimum
        """
        return numpy.min(y)
    
    def extract_time_min(self, x, y):
        """ Extracts the time at the minimum of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            time at the minimum
        """
        return x[numpy.argmin(y)]
    
    def extract_max(self, x, y):
        """ Extracts the maximum of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            maximum
        """
        return numpy.max(y)
    
    def extract_time_max(self, x, y):
        """ Extracts the time at the maximum of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            time at the maximum
        """
        return x[numpy.argmax(y)]
    
    def extract_span(self, x, y):
        """ Extracts the span between minimum and maximum.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            span
        """
        return numpy.max(y) - numpy.min(y)
    
    def extract_median(self, x, y):
        """ Extracts the median of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            median
        """
        return numpy.median(y)
    
    def extract_stan_dev(self, x, y):
        """ Extracts the standard deviation of y.

        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            standard deviation
        """
        return numpy.std(y)


class DOFeatureExtractor(Extractor):
    """ Class for feature extraction of dissolved oxygen"""
    
    def extract_peak(self, x, y):
        """ Extracts the dissolved oxygen limitation.
        
        Returns
        -------
        result : float
            time spans of all values under 5
        """
        tmp = 0
        boolean = False
        time = 0
        #if values are under 5, the time spans are calculated and added together
        for i in range(len(y)):
            if y[i] < 5 and boolean is False:
                boolean = True
                tmp = i
            if y[i] > 5 and boolean is True: 
                boolean = False
                time = time + x[i] - x[tmp]
        if boolean is True:
            time = time + x[len(y)-1] - x[tmp]
        return time
    
    
class BSFeatureExtractor(Extractor):
    """ Class for feature extraction of backscatter. """
    
    def extract_turning_point(self, x, y):
        """ Extracts value at the turning point.
        
        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            turning point
        """
        spline = bletl_analysis.get_crossvalidated_spline(x, y)
        y_der1 = spline.derivative(1)(x)
        return y[numpy.argmax(y_der1)]
    
    def extract_time_turning_point(self, x, y):
        """ Extracts time at the turning point.
        
        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            time of turning point
        """
        spline = bletl_analysis.get_crossvalidated_spline(x, y)
        y_der1 = spline.derivative(1)(x)
        return x[numpy.argmax(y_der1)]
    
    def extract_mue_median(self,x ,y):
        """ Extracts median of mue at the exponential phase.
        
        Parameters
        ----------
        x : []
            list of time values
        y : []
            list of values
        
        Returns
        -------
        result : float
            median
        """
        #calculate value, which lies in exponential phase
        mid_y = (numpy.max(y) - numpy.min(y)) / 2
        #get index of a value, which lies in exponential phase
        mid_i = int(numpy.where(y > mid_y)[0][0]/2)
        #get mue
        bsdata = bletl.core.FilterTimeSeries(pandas.DataFrame(x),pandas.DataFrame(y))
        mue = bletl_analysis.get_mue(bsdata, method='us')
        var = 0.1 #allowed variance
        mue_val = numpy.array(mue.value) 
        list_values = [mue_val[mid_i]] #list of mue values, which lie in ex. phase 
        l_i = mid_i - 1
        r_i = mid_i + 1
        #checks if mue_val[l_i] lies in variance of mue_val from mid_i, goes left
        while l_i >= 0 and mue_val[l_i] < mue_val[mid_i] + var and mue_val[l_i] > mue_val[mid_i] - var:
            list_values.append(mue_val[l_i])
            l_i = l_i - 1
        #checks if mue_val[l_i] lies in variance of mue_val from mid_i, goes right
        while r_i < len(mue_val) and mue_val[r_i] < mue_val[mid_i] + var and mue_val[r_i] > mue_val[mid_i] - var:
            list_values.append(mue_val[r_i])     
            r_i = r_i + 1
        return numpy.median(list_values)


class pHFeatureExtractor(Extractor):
    """ Class for feature extraction of pH."""
    
    
class TSFreshExtractor():
    """ Class for feature extraction with tsfresh."""
    
    def _extract(self, data):
        """ Extracts data with tsfresh.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame of a filterset
        
        Returns
        -------
        result : pandas.DataFrame
            DataFrame with extracted features
        """
        return tsfresh.extract_features(data, column_id="id", column_sort="time").T
    
    
def from_bldata(
    bldata: bletl.BLData,
    extractors: typing.Dict[str, typing.Sequence[Extractor]],
    last_cycles: typing.Optional[typing.Dict[str, int]]=None,
) -> pandas.DataFrame:
    """ Apply feature extractors to a dataset.

    Parameters
    ----------
    data : bletl.BLData
        a dataset to extract from
    extractors : dict
        map of { filterset : [extractor, ...] }
    last_cycles :  optional, dict
        maps well ids to the number of the last cycle to consider
        
    Returns
    -------
    result : pandas.DataFrame
        well-indexed features
    """
    # create useful local variables from inputs
    if not last_cycles:
        last_cycles = {}
    filtersets = set(extractors.keys())
    wells = tuple(sorted(bldata[filtersets[0]].value.columns))
    for fs in filtersets.difference(set(data.keys())):
        _log.warning('No "%s" filterset in the data. Skipping extractors.', fs)
    extraction_methods = {
        f'{fs}_{mname}': method
        for fs, methods in extractors.items()
        for mname, method in methods.items()
    }

    _log.info("Extracting from %i wells.", len(wells))
    s_time = time.time()
    df_result = pandas.DataFrame(index=wells)
    for well in fastprogress.progress_bar(wells):
        t, y = bldata[fs].get_timeseries(well, last_cycle=last_cycles.get(well, d=None))
        for mname, method in extractextraction_methods.items():
            df_results.loc[well, mname] = method(t, y)
    _log.info("Extraction finished in: %s minutes", round((time.time() - s_time) / 60, ndigits=1))
    return df_result
