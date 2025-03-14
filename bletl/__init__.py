# bletl
# Copyright (C) 2019  Forschungszentrum Jülich GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# For more information contact the maintainers of https://github.com/JuBiotech.
"""BioLector Extraction Transformation and Loading (bletl) is a package for parsing raw
BioLector files, applying calibration transformations and representing them in a standardized
format.
"""
import importlib.metadata

from . import utils
from .core import get_parser, parse, parsers
from .heuristics import find_do_peak
from .splines import get_crossvalidated_spline
from .types import (
    BioLectorModel,
    BLData,
    BLDParser,
    FilterTimeSeries,
    FluidicsSource,
    IncompatibleFileError,
    InvalidLotNumberError,
    LotInformationError,
    LotInformationMismatch,
    LotInformationNotFound,
    NoMeasurementData,
)

__version__ = importlib.metadata.version(__package__ or __name__)
