"""BioLector Extraction Transformation and Loading (bletl) is a package for parsing raw
BioLector files, applying calibration transformations and representing them in a standardized
format.
"""
import pandas
import urllib.request
import configparser

from . core import BioLectorModel, BLData, BLDParser
from . import parsing
from . import utils

__version__ = '0.5'

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


def parse(filepath, drop_incomplete_cycles:bool=True) -> BLData:
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


def parse_and_concatenate(filepaths:list, drop_incomplete_cycles:bool=True) -> BLData:
    """Parses multiple BioLector raw data files and concatenates them into one.

    Args:
        filepaths (list): list of filepaths. Files should be in chronological order.
        drop_incomplete_cycles (bool): if True, incomplete cycles at the end are discarded
            IMPORTANT: all fragments should have at least one FULL cycle!

    Returns:
        bldata (BLData): object containing all the measurements, as if they would have
            originated from the same file
    """
    fragments = [
        parse(filepath)
        for filepath in filepaths
    ]
    head = fragments[0]

    # iterate over all DataFrame-attributes
    for attr, stack in head.__dict__.items():
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
            setattr(head, attr, stack)

    head.metadata['date_end'] = fragments[-1].metadata['date_end']

    return head


def get_calibration_dict(lot_number:int, temp:int) -> dict:
    """Loads calibration data from M2P-labs website

    Args:
        lot_number (int): Lot number of plate to be used for calibration.
        temp (int): Temperature to be used for calibration.

    Returns:
        calibration_dict (dict): Dictionary containing calibration data. 
            Can be readily used in calibration function.
    """
    lookup_string = f"{lot_number}-hc-Temp{temp}"
    url = 'http://updates.m2p-labs.com/CalibrationLot.ini'
    content = urllib.request.urlopen(url).read().decode()

    parser = configparser.ConfigParser(strict=False)
    parser.read_string(content)

    calibration_dict = {
        'cal_0': float(parser[lookup_string]['k0']),
        'cal_100': float(parser[lookup_string]['k100']),
        'phi_min': float(parser[lookup_string]['irmin']),
        'phi_max': float(parser[lookup_string]['irmax']),
        'pH_0': float(parser[lookup_string]['ph0']),
        'dpH': float(parser[lookup_string]['dph']),
    }

    return calibration_dict
