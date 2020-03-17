"""BioLector Extraction Transformation and Loading (bletl) is a package for parsing raw
BioLector files, applying calibration transformations and representing them in a standardized
format.
"""
import pandas
import urllib.request
import urllib.error
import configparser
import pathlib
from collections.abc import Iterable

from . core import BioLectorModel, BLData, BLDParser, LotInformationError, LotInformationMismatch, InvalidLotNumberError, LotInformationNotFound, FilterTimeSeries
from . import parsing
from . import utils

__version__ = '0.12.1'

parsers = {
    (BioLectorModel.BL1, '3.3') : parsing.bl1.BioLector1Parser,
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
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    model = None
    version = None

    if lines[0].startswith('='):
        model = BioLectorModel.BLPro
        for line in lines:
            if line.startswith('[file_version_number]'):
                version = line.split(' ')[1].strip()
                break
    elif lines[0].startswith('FILENAME;'):
        # TODO: detect logfiles from BioLector 2
        model = BioLectorModel.BL1
        version = lines[2][13:-2]
    else:
        raise NotImplementedError('Unsupported file version')

    # select a parser for this version
    parser_cls = parsers[(model, version)]
    return parser_cls()


def _parse_without_calibration(filepath:str, drop_incomplete_cycles:bool) -> BLData:
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
    data = parser.parse(filepath)

    if drop_incomplete_cycles:
        index_names, measurements = utils._unindex(data.measurements)
        latest_full_cycle = utils._last_full_cycle(measurements)
        measurements = measurements[measurements.cycle <= latest_full_cycle]
        data.measurements = utils._reindex(measurements, index_names)

    return data


def _apply_calibration(data:BLData, lot_number:int=None, temp:int=None) -> BLData:
    """Applies calibration to an BLdata object.

    Args:
        data (BLdata): BLdata object to calibrate
        lot_number (int or None): lot number of the microtiter plate used
        temp (int or None): Temperature to be used for calibration

    Returns:
        BLData: calibrated data object

    Raises:
        TypeError: when either lot number or temperature, but not both, are None
        NotImplementedError: when the file contents do not match with a known BioLector CSV style
        LotInformationError: when no information about the lot can be found
        LotInformationMismatch: when lot information given as parameters is not equal to lot information found in data file
    """
    if data.model is BioLectorModel.BL1:
        if (lot_number is None) ^ (temp is None):
            raise TypeError('Lot number and temperature should be either left None or be set to an appropriate value.')
        
        if (lot_number is None) and (temp is None):
            if (data.metadata['lot'] in {'UNKNOWN', 'UNKOWN'}):
                data.calibrate()
            else:
                cal_data = fetch_calibration_data(*utils._parse_calibration_info(data.metadata['lot']))
                if cal_data is None:
                    raise LotInformationNotFound("""Lot information was found in the CSV file,
                        but the calibration data was not found in the cache and the cache could not be updated.
                        No calibration for pH and DO is applied.""")
                data.calibrate(cal_data)

        if isinstance(lot_number, int) and isinstance(temp, int):
            if not (data.metadata['lot'] in {'UNKNOWN', 'UNKOWN'}):
                lot_from_csv, temp_from_csv = utils._parse_calibration_info(data.metadata['lot'])
                if (lot_number != lot_from_csv) or (temp != temp_from_csv):
                    raise LotInformationMismatch('The lot information provided mismatches with lot information found in the data file.\
                        The provided lot information is used for calibration.')
            cal_data = fetch_calibration_data(lot_number, temp)
            if cal_data is None:
                raise LotInformationError('Data for the lot information provided was not found in the cached file \
                    and we were unable to update it. If you want to proceed without calibration, pass no lot number and temperature')
            data.calibrate(cal_data)
                
    return data


def parse(filepaths, lot_number:int=None, temp:int=None, drop_incomplete_cycles:bool=True) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object and applies calibration.

    Args:
        filepaths (str or pathlib.Path or iterable): path pointing to the file(s) of interest. If an iterable is provided, files are concatenated
        lot_number (int or None): lot number of the microtiter plate used
        temp (int or None): Temperature to be used for calibration
        drop_incomplete_cycles (bool): if True, incomplete cycles at the end are discarded
            IMPORTANT: if the file contains only one cycle, it will always be considered "completed"

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
            fragment = _parse_without_calibration(filepath, drop_incomplete_cycles)
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
        data = _parse_without_calibration(filepaths, drop_incomplete_cycles)
    
    data = _apply_calibration(data, lot_number, temp)

    return data


def parse_with_calibration_parameters(
    filepaths,
    cal_0:float=None,
    cal_100:float=None,
    phi_min:float=None,
    phi_max:float=None,
    pH_0:float=None,
    dpH:float=None,
    drop_incomplete_cycles:bool=True,
    ) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object while using explicit calibration parameters.

    Args:
        filepaths (str or pathlib.Path or iterable): path pointing to the file(s) of interest. If an iterable is provided, files are concatenated
        cal_0 (float or None): Calibration parameter cal_0 or k0 for oxygen saturation measurement
        cal_100 (float or None): Calibration parameter cal_100 or k100 for oxygen saturation measurement
        phi_min (float or None): Calibration parameter phi_min or irmin for pH measurement
        phi_max (float or None): Calibration parameter phi_max or irmax for pH measurement
        pH_0 (float or None): Calibration parameter ph0 for pH measurement
        dpH (float or None): Calibration parameter dpH for pH measurement
        drop_incomplete_cycles (bool): if True, incomplete cycles at the end are discarded
            IMPORTANT: if the file contains only one cycle, it will always be considered "completed"

    Returns:
        BLData: parsed data object

    Raises:
        NotImlementedError: when the file contents do not match with a known BioLector CSV style
    """
    calibration_dict = {
        'cal_0': cal_0,
        'cal_100': cal_100,
        'phi_min': phi_min,
        'phi_max': phi_max,
        'pH_0': pH_0,
        'dpH': dpH,
    }

    if isinstance(filepaths, Iterable):
        fragments = []
        for filepath in filepaths:
            fragment = _parse_without_calibration(filepath, drop_incomplete_cycles)
            fragments.append(fragment)
        
        data = fragments[0]

        # iterate over all DataFrame-attributes
        for attr, stack in data.__dict__.items():
            if isinstance(stack, pandas.DataFrame):
                # time/cycle aware concatenation of all fragments
                fragment_frames = [
                    getattr(fragment, attr)
                    for fragment in fragments
                ]
                start_times = [
                    fragment.metadata['date_start']
                    for fragment in fragments
                ]
                stack = utils._concatenate_fragments(fragment_frames, start_times)
                setattr(data, attr, stack)

        data.metadata['date_end'] = fragments[-1].metadata['date_end']
    else:
        data = _parse_without_calibration(filepaths, drop_incomplete_cycles)

    data.calibrate(calibration_dict)

    return data


def fetch_calibration_data(lot_number:int, temp:int):
    """Loads calibration data from M2P-labs website

    Args:
        lot_number (int): Lot number to be used for calibration data lookup
        temp (int): Temperature to be used for calibration data lookup

    Returns:
        calibration_dict (dict): Dictionary containing calibration data.
            Can be readily used in calibration function.
        None (None): 
    """
    module_path = __path__[0]
    calibration_file = pathlib.Path(module_path, 'cache', 'CalibrationLot.ini')

    if not calibration_file.is_file():
        if not download_calibration_data():
            return None

    with open(calibration_file, 'r') as file:
        content = file.read()

    parser = configparser.ConfigParser(strict=False)
    parser.read_string(content)
    calibration_info = f'{lot_number}-hc-Temp{temp}'

    if not calibration_info in parser:
        if not download_calibration_data():
            return None
    
    if calibration_info in parser:
        calibration_dict = {
            'cal_0': float(parser[calibration_info]['k0']),
            'cal_100': float(parser[calibration_info]['k100']),
            'phi_min': float(parser[calibration_info]['irmin']),
            'phi_max': float(parser[calibration_info]['irmax']),
            'pH_0': float(parser[calibration_info]['ph0']),
            'dpH': float(parser[calibration_info]['dph']),
        }
        return calibration_dict
    else:
        raise InvalidLotNumberError("""Latest calibration information was downloaded from m2p-labs, 
            but the provided lot number/temperature combination could not be found. Please check the parameters.""")

def download_calibration_data():
    """Loads calibration data from m2p-labs website

    Returns:
        success (bool): True if calibration data was downloaded successfully, False otherwise
    """
    try:
        url = 'http://updates.m2p-labs.com/CalibrationLot.ini'
        module_path = __path__[0]
        filepath = pathlib.Path(module_path, 'cache', 'CalibrationLot.ini')
        filepath.parents[0].mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        return True
    
    except urllib.error.HTTPError:
        return False
    
    