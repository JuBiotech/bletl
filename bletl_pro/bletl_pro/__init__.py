"""BioLector Extraction Transformation and Loading (bletl) is a package for parsing raw
BioLector files, applying calibration transformations and representing them in a standardized
format.

bletl_pro adds functionality for working with BioLector Pro files.
"""
import bletl
from bletl import *
from . parsing import blpro

__version__ = '0.3.3'

# register the pro-parser
bletl.parsers[(BioLectorModel.BLPro, '0.0.0')] = parsing.blpro.BioLectorProParser
