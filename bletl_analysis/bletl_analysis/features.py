import abc
import inspect
import logging
import numpy
import pandas
import time
import typing

import bletl


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
