"""Specifies the base types for parsing and representing BioLector CSV files."""
import abc
from collections.abc import Iterable
import enum
import numpy
import pandas
import pathlib
import typing
import urllib.request
import urllib.error
import warnings

from . import parsing
from . types import BLData, BLDParser, FilterTimeSeries, BioLectorModel, LotInformationError, InvalidLotNumberError, LotInformationMismatch, LotInformationNotFound, IncompatibleFileError, NoMeasurementData
from . import utils


parsers = {
    (BioLectorModel.BL1, '3.3') : parsing.bl1.BioLector1Parser,
    (BioLectorModel.BLPro, '0.0.0') : parsing.blpro.BioLectorProParser,
}


def get_parser(filepath) -> BLDParser:
    """Analyzes a raw BioLector file and selects an appropiate parser.

    Args:
        filepath (str or pathlib.Path): path pointing to the file of interest

    Returns:
        BLDParser: a parser that can be used for the provided file type

    Raises:
        NotImlementedError: when the file contents do not match with a known BioLector CSV style
    """
    try:
        # Note:
        # BioLector II files are encoded as UTF8-BOM
        # BioLector Pro files are encoded as UTF-8
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    model = None
    version = None

    if '=====' in lines[0]:
        if 'parameters' in lines[0]:
            raise IncompatibleFileError('It seems like this file has been edited by the BioLection software. Please provide the raw data file.')
        model = BioLectorModel.BLPro
        for line in lines:
            if line.startswith('[file_version_number]'):
                version = line.split(']')[1].strip()
                break
    elif 'FILENAME;' in lines[0]:
        model = BioLectorModel.BL1
        version = lines[2][13:-2]
    if not model:
        raise ValueError('Unable to detect the BioLector model from the file contents.')
    if not version :
        raise ValueError('Unable to detect the file version from the contents.')
    if not (model, version) in parsers:
        raise NotImplementedError(f'Unsupported {model} file version: {version}')

    # select a parser for this version
    parser_cls = parsers[(model, version)]
    return parser_cls()


def _parse(filepath:str, lot_number:int, temp:int, drop_incomplete_cycles:bool, 
    cal_0:float=None, cal_100:float=None, phi_min:float=None, phi_max:float=None, pH_0:float=None, dpH:float=None
) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object.

    Args:
        filepath (str or pathlib.Path): path pointing to the file of interest
        drop_incomplete_cycles (bool): if True, incomplete cycles at the end are discarded
            IMPORTANT: if the file contains only one cycle, it will always be considered "completed"

    Returns:
        BLData: parsed data object

    Raises:
        NotImlementedError: when the file contents do not match with a known BioLector CSV style
    """ 
    parser = get_parser(filepath)
    data = parser.parse(filepath, lot_number, temp, cal_0, cal_100, phi_min, phi_max, pH_0, dpH)

    if (not data.measurements.empty) and drop_incomplete_cycles:
        index_names, measurements = utils._unindex(data.measurements)
        latest_full_cycle = utils._last_full_cycle(measurements)
        measurements = measurements[measurements.cycle <= latest_full_cycle]
        data._measurements = utils._reindex(measurements, index_names)

    return data


def parse(
    filepaths, lot_number:int=None, temp:int=None, drop_incomplete_cycles:bool=True, 
    cal_0:float=None, cal_100:float=None, phi_min:float=None, phi_max:float=None, pH_0:float=None, dpH:float=None
) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object and applies calibration.

    Args:
        filepaths (str or pathlib.Path or iterable): path pointing to the file(s) of interest. If an iterable is provided, files are concatenated
        lot_number (int or None): lot number of the microtiter plate used
        temp (int or None): Temperature to be used for calibration
        drop_incomplete_cycles (bool): if True, incomplete cycles at the end are discarded
            IMPORTANT: if the file contains only one cycle, it will always be considered "completed"
        cal_0 (float or None): Calibration parameter cal_0 or k0 for oxygen saturation measurement
        cal_100 (float or None): Calibration parameter cal_100 or k100 for oxygen saturation measurement
        phi_min (float or None): Calibration parameter phi_min or irmin for pH measurement
        phi_max (float or None): Calibration parameter phi_max or irmax for pH measurement
        pH_0 (float or None): Calibration parameter ph0 for pH measurement
        dpH (float or None): Calibration parameter dpH for pH measurement
    Returns:
        BLData: parsed data object

    Raises:
        TypeError: when either lot number or temperature, but not both, are None
        NotImplementedError: when the file contents do not match with a known BioLector CSV style
        LotInformationError: when no information about the lot can be found
        LotInformationMismatch: when lot information given as parameters is not equal to lot information found in data file
    """
    if isinstance(filepaths, Iterable) and not isinstance(filepaths, str):
        fragments = []
        for filepath in filepaths:
            fragment = _parse(filepath, lot_number, temp, drop_incomplete_cycles, cal_0, cal_100, phi_min, phi_max, pH_0, dpH)
            fragments.append(fragment)
        start_times = [
            fragment.metadata['date_start']
            for fragment in fragments
        ]
        
        data = fragments[0]

        # iterate over all DataFrame-attributes
        for attr, stack in data.__dict__.items():
            if isinstance(stack, pandas.DataFrame):
                # time/cycle aware concatenation of all fragments
                fragment_frames = [
                    getattr(fragment, attr)
                    for fragment in fragments
                ]
                stack = utils._concatenate_fragments(fragment_frames, start_times)
                setattr(data, attr, stack)
        
        # also iterate over FilterTimeSeries and concatenate them
        if len(fragments) > 1:
            for fs in data.keys():
                # already increment the time here, because utils._concatenate_fragments won't do that
                conc_times = utils._concatenate_fragments([
                   f[fs].time + (fragment_start - start_times[0]).total_seconds() / 3600
                   for f, fragment_start in zip(fragments, start_times)
                ], start_times)
                conc_values = utils._concatenate_fragments([
                    f[fs].value
                    for f in fragments
                ], start_times)
                # overwrite with concatenated FilterTimeSeries
                data[fs] = FilterTimeSeries(conc_times, conc_values)

        data.metadata['date_end'] = fragments[-1].metadata['date_end']
    else:
        data = _parse(filepaths, lot_number, temp, drop_incomplete_cycles, cal_0, cal_100, phi_min, phi_max, pH_0, dpH)

    return data