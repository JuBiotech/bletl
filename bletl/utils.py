"""Contains helper functions that do not depend on other modules within this package."""

import datetime
import enum
import pathlib
import re
import urllib
import urllib.error
import urllib.request
from typing import Optional, Sequence, Tuple

import pandas

from . import core


def __to_typed_cols__(
    dfin: pandas.DataFrame, ocol_ncol_type: Sequence[Tuple[Optional[str], str, type]]
) -> pandas.DataFrame:
    """Can be used to filter & convert data frame columns.

    Parameters
    ----------
    dfin : pandas.DataFrame
        Raw data frame to start from.
    ocol_ncol_type : list of tuples
        Maps original to new column names and desired data types.
        Entries should be in the form `('original column', 'new column', datatype)`.

    Returns
    -------
    dfout : DataFrame
        A new data frame with converted & renamed columns as specified by `ocol_ncol_type`.
    """
    dfout = pandas.DataFrame()
    for ocol, ncol, typ in ocol_ncol_type:
        if ocol is None or not ocol in dfin:
            dfout[ncol] = None
        elif issubclass(typ, enum.Enum):
            # Enum types are kept as object-series
            dfout[ncol] = pandas.Series([typ(x) for x in dfin[ocol]], name=ncol, dtype=object)
        else:
            dfout[ncol] = dfin[ocol].astype(typ)
    return dfout


def _unindex(dataframe: pandas.DataFrame) -> Tuple[Sequence[Optional[str]], pandas.DataFrame]:
    """Resets the index of the DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Guess what.

    Returns
    -------
    index_names : FrozenList
        All the index names. May be (None,).
    dataframe : pandas.DataFrame
        Frame without indices
    """
    return dataframe.index.names, dataframe.reset_index()


def _reindex(dataframe: pandas.DataFrame, index_names: Sequence[str]) -> pandas.DataFrame:
    """Applies an indexing scheme to a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Guess what.
    index_names : tuple of str
        All the index names. May be (None,).

    Returns
    -------
    dataframe : pandas.DataFrame
        Frame with the indexing scheme.
    """
    if index_names[0] is not None:
        return dataframe.set_index(index_names)
    else:
        return dataframe


def _concatenate_fragments(
    fragments: Sequence[pandas.DataFrame], start_times: Sequence[datetime.datetime]
) -> pandas.DataFrame:
    """Concatenate multiple dataframes while shifting time and cycles.

    Parameters
    ----------
    fragments : list of dataframes
        DataFrames to concatenate.
    start_times : list of datetimes
        Experiment start times for each data fragment.

    Returns
    -------
    concatenation : pandas.DataFrame
        Time/cycle-aware concatenation of fragments.
    """
    index_names, stack = _unindex(fragments[0])
    columns = set(stack.columns)

    for fragment, fragment_start in zip(fragments[1:], start_times[1:]):
        assert isinstance(fragment, pandas.DataFrame), "fragments must be a list of DataFrames"
        index_names_f, fragment = _unindex(fragment)
        assert set(index_names_f) == set(index_names), "indices must match across all fragments"
        assert set(fragment.columns) == columns, "columns must match across all fragments"

        # shift time and cycle columns in the fragment
        if "time" in columns:
            fragment["time"] += (fragment_start - start_times[0]).total_seconds() / 3600
        if "cycle" in columns and len(stack) > 0:
            fragment["cycle"] += max(stack["cycle"])

        # append the fragment to the stack
        stack = pandas.concat((stack, fragment))

    # re-apply the original indexing scheme
    return _reindex(stack, index_names)  # type: ignore


def _last_well_in_cycle(measurements: pandas.DataFrame) -> Optional[str]:
    """Finds the name of the last well measured in a cycle.

    Parameters
    ----------
    measurements : pandas.DataFrame
        Measurements data.

    Returns
    -------
    well : str
        Name of the last well measured in the first cycle.
        If the cycle was incomplete, the last measured well is returned!
    """
    previous_well = None
    previous_cycle = None
    for cycle, well in zip(measurements.cycle, measurements.well):
        if previous_cycle and cycle > previous_cycle:
            return previous_well
        else:
            previous_cycle, previous_well = cycle, well
    return previous_well


def _last_full_cycle(measurements: pandas.DataFrame) -> int:
    """Find the number of the last cycle that was measured for all wells and filters.

    Parameters
    ----------
    measurements : pandas.DataFrame
        Measurements data

    Returns
    -------
    last_cycle : int
        Number of the last complete cycle.
        NOTE: if the data contains only one cycle, it will always be considered "completed".
    """
    max_filter = max(measurements.filterset)
    max_well = _last_well_in_cycle(measurements)

    last_filter = measurements.iloc[-1].filterset
    last_cycle = measurements.iloc[-1].cycle
    last_well = measurements.iloc[-1].well

    if (last_filter, last_well) != (max_filter, max_well):
        return last_cycle - 1
    else:
        return last_cycle


def _parse_calibration_info(calibration_info: str):
    """Extracts lot number and temperature for a line of calibration info.

    Parameters
    ----------
    calibration_info : str
        Calibration info e. g. from CSV file such as '1818-hc-Temp30'.

    Returns
    -------
    lot_number : int
        Lot number
    temp : int
        Process temperature
    """
    result = re.findall(r"(\d*)-hc-Temp(\d{2})", calibration_info)
    lot_number = int(result[0][0])
    temp = int(result[0][1])

    return lot_number, temp


def download_calibration_data() -> bool:
    """Loads calibration data from m2p-labs website

    Returns
    -------
    success : bool
        `True` if calibration data was downloaded successfully, `False` otherwise.
    """
    try:
        assert core.__spec__ is not None
        assert core.__spec__.origin is not None
        module_path = pathlib.Path(core.__spec__.origin).parents[0]

        url_bl1 = "http://updates.m2p-labs.com/CalibrationLot.ini"
        filepath_bl1 = pathlib.Path(module_path, "cache", "CalibrationLot.ini")
        filepath_bl1.parents[0].mkdir(exist_ok=True)
        urllib.request.urlretrieve(url_bl1, filepath_bl1)

        url_blpro = "http://updates.m2p-labs.com/CalibrationLot_II.xml"
        filepath_blpro = pathlib.Path(module_path, "cache", "CalibrationLot_II.xml")
        filepath_blpro.parents[0].mkdir(exist_ok=True)
        urllib.request.urlretrieve(url_blpro, filepath_blpro)

        return True

    except urllib.error.HTTPError:
        return False
