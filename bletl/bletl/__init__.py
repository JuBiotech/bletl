"""BioLector Extraction Transformation and Loading (bletl) is a package for parsing raw
BioLector files, applying calibration transformations and representing them in a standardized
format.
"""
from . core import BioLectorModel, BLData, BLDParser
from . import parsing

__version__ = '0.2'

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

def parse(filepath) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object.

    Args:
        filepath (str or pathlib.Path): path pointing to the file of interest

    Returns:
        BLData: parsed data object

    Raises:
        NotImlementedError: when the file contents do not match with a known BioLector CSV style
    """
    parser = get_parser(filepath)
    return parser.parse(filepath)

