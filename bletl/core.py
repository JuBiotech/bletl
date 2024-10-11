"""Specifies the base types for parsing and representing BioLector CSV files."""
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas

from . import parsing, utils
from .types import (
    BioLectorModel,
    BLData,
    BLDParser,
    FilterTimeSeries,
    IncompatibleFileError,
)

parsers = {
    (BioLectorModel.BL1, "3.3"): parsing.bl1.BioLector1Parser,
    (BioLectorModel.BLPro, "0.0.0"): parsing.blpro.BioLectorProParser,
}


def get_parser(filepath: Union[str, Path]) -> BLDParser:
    """Analyzes a raw BioLector file and selects an appropiate parser.

    Parameters
    ----------
    filepath : str or Path
        Path pointing to the file of interest.

    Returns
    -------
    parser : BLDParser
        A parser that can be used for the provided file type.

    Raises
    ------
    NotImlementedError
        When the file contents do not match with a known BioLector refult file format.
    """

    model = None
    version = None

    # Check for XT file types first because they are zipped
    if Path(filepath).suffix.lower() == ".zip":
        with zipfile.ZipFile(filepath, "r") as zfile:
            for fp in zfile.namelist():
                if not fp.endswith(".meta"):
                    continue
                with zfile.open(fp) as jfile:
                    metadict = json.load(jfile)
                version = metadict["CsvFileVersion"]
                model = BioLectorModel.XT
                if not (model, version) in parsers:
                    raise NotImplementedError(f"Unsupported {model} file version: {version}")
                return parsers[(model, version)]()
        raise IncompatibleFileError("Unable to detect the BioLector model from the file contents.")

    # Now check I/II/Pro file which are plain text
    try:
        # Note:
        # BioLector II files are encoded as UTF8-BOM
        # BioLector Pro files are encoded as UTF-8
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            lines = f.readlines()

    if "=====" in lines[0]:
        if "parameters" in lines[0]:
            raise IncompatibleFileError(
                "It seems like this file has been edited by the BioLection software. Please provide the raw data file."
            )
        model = BioLectorModel.BLPro
        for line in lines:
            if line.startswith("[file_version_number]"):
                version = line.split("]")[1].strip()
                break
    elif "FILENAME;" in lines[0]:
        model = BioLectorModel.BL1
        version = lines[2][13:-2]
    if not model:
        raise ValueError("Unable to detect the BioLector model from the file contents.")
    if not version:
        raise ValueError("Unable to detect the file version from the contents.")
    if not (model, version) in parsers:
        raise NotImplementedError(f"Unsupported {model} file version: {version}")

    # select a parser for this version
    parser_cls = parsers[(model, version)]
    return parser_cls()


def _parse(
    filepath: Union[str, Path],
    drop_incomplete_cycles: bool,
    lot_number: Optional[int],
    temp: Optional[int],
    cal_0: Optional[float] = None,
    cal_100: Optional[float] = None,
    phi_min: Optional[float] = None,
    phi_max: Optional[float] = None,
    pH_0: Optional[float] = None,
    dpH: Optional[float] = None,
) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object.

    Parameters
    ----------
    filepath : str or Path
        Path pointing to the file of interest.
    drop_incomplete_cycles : bool
        If `True`, incomplete cycles at the end are discarded.
        IMPORTANT: if the file contains only one cycle, it will always be considered "completed".
    lot_number : int or None
        Lot number of the microtiter plate used.
    temp :int, optional
        Temperature to be used for calibration.
    cal_0 : float, optional
        Calibration parameter cal_0 or k0 for oxygen saturation measurement.
    cal_100 : float, optional
        Calibration parameter cal_100 or k100 for oxygen saturation measurement.
    phi_min : float, optional
        Calibration parameter phi_min or irmin for pH measurement.
    phi_max : float, optional
        Calibration parameter phi_max or irmax for pH measurement.
    pH_0 : float, optional
        Calibration parameter ph0 for pH measurement.
    dpH : float, optional
        Calibration parameter dpH for pH measurement.

    Returns
    -------
    bldata : BLData
        Parsed data object.

    Raises
    ------
    NotImlementedError
        When the file contents do not match with a known BioLector result file format.
    """
    parser = get_parser(filepath)
    data = parser.parse(
        filepath,
        lot_number=lot_number,
        temp=temp,
        cal_0=cal_0,
        cal_100=cal_100,
        phi_min=phi_min,
        phi_max=phi_max,
        pH_0=pH_0,
        dpH=dpH,
    )

    if (not data.measurements.empty) and drop_incomplete_cycles:
        index_names, measurements = utils._unindex(data.measurements)
        latest_full_cycle = utils._last_full_cycle(measurements)
        measurements = measurements[measurements.cycle <= latest_full_cycle]
        data._measurements = utils._reindex(measurements, index_names)  # type: ignore

    return data


def parse(
    filepaths: Union[Union[str, Path], Sequence[Union[str, Path]]],
    *,
    drop_incomplete_cycles: bool = True,
    lot_number: Optional[int] = None,
    temp: Optional[int] = None,
    cal_0: Optional[float] = None,
    cal_100: Optional[float] = None,
    phi_min: Optional[float] = None,
    phi_max: Optional[float] = None,
    pH_0: Optional[float] = None,
    dpH: Optional[float] = None,
) -> BLData:
    """Parses a raw BioLector CSV file into a BLData object and applies calibration.

    Parameters
    ----------
    filepaths : str or Path or iterable
        Path pointing to the file(s) of interest.
        If an iterable is provided, files are concatenated.
    drop_incomplete_cycles : bool
        If `True`, incomplete cycles at the end are discarded
        IMPORTANT: if the file contains only one cycle, it will always be considered "completed".
    lot_number : int or None
        Lot number of the microtiter plate used.
    temp :int, optional
        Temperature to be used for calibration.
    cal_0 : float, optional
        Calibration parameter cal_0 or k0 for oxygen saturation measurement.
    cal_100 : float, optional
        Calibration parameter cal_100 or k100 for oxygen saturation measurement.
    phi_min : float, optional
        Calibration parameter phi_min or irmin for pH measurement.
    phi_max : float, optional
        Calibration parameter phi_max or irmax for pH measurement.
    pH_0 : float, optional
        Calibration parameter ph0 for pH measurement.
    dpH : float, optional
        Calibration parameter dpH for pH measurement.

    Returns
    -------
    bldata : BLData
        Parsed data object.

    Raises
    ------
    TypeError
        When either lot number or temperature, but not both, are None.
    NotImplementedError
        When the file contents do not match with a known BioLector result file format.
    LotInformationError
        When no information about the lot can be found.
    LotInformationMismatch
        When lot information given as parameters is not equal to lot information found in data file.
    """
    if isinstance(filepaths, Iterable) and not isinstance(filepaths, str):
        fragments = []
        for filepath in filepaths:
            fragment = _parse(
                filepath,
                drop_incomplete_cycles,
                lot_number,
                temp,
                cal_0,
                cal_100,
                phi_min,
                phi_max,
                pH_0,
                dpH,
            )
            fragments.append(fragment)
        start_times = [fragment.metadata["date_start"] for fragment in fragments]

        data = fragments[0]

        # iterate over all DataFrame-attributes
        for attr, stack in data.__dict__.items():
            if isinstance(stack, pandas.DataFrame):
                # time/cycle aware concatenation of all fragments
                fragment_frames = [getattr(fragment, attr) for fragment in fragments]
                stack = utils._concatenate_fragments(fragment_frames, start_times)
                setattr(data, attr, stack)

        # also iterate over FilterTimeSeries and concatenate them
        if len(fragments) > 1:
            for fs in data.keys():
                # already increment the time here, because utils._concatenate_fragments won't do that
                conc_times = utils._concatenate_fragments(
                    [
                        f[fs].time + (fragment_start - start_times[0]).total_seconds() / 3600
                        for f, fragment_start in zip(fragments, start_times)
                    ],
                    start_times,
                )
                conc_values = utils._concatenate_fragments([f[fs].value for f in fragments], start_times)
                # overwrite with concatenated FilterTimeSeries
                data[fs] = FilterTimeSeries(conc_times, conc_values)

        data.metadata["date_end"] = fragments[-1].metadata["date_end"]
    else:
        data = _parse(
            filepaths, drop_incomplete_cycles, lot_number, temp, cal_0, cal_100, phi_min, phi_max, pH_0, dpH
        )

    return data
