"""Parsing functions for the BioLector Pro"""
import collections
import datetime
import io
import logging
import os
import pathlib
import re
import warnings
import xml.etree.ElementTree
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy
import pandas

from .. import utils
from ..types import (
    BioLectorModel,
    BLData,
    BLDParser,
    FilterTimeSeries,
    FluidicsSource,
    IncompatibleFileError,
    InvalidLotNumberError,
)

logger = logging.getLogger("blpro")


_MF_WELL_NUMC_TO_ID = {
    (colnumber + rownumber * 8): f"{row}{column:02d}"
    for colnumber, column in enumerate(range(1, 9))
    for rownumber, row in enumerate("CDEF")
}
"""Maps 0-based well numbers in MF mode by **C-style** counting order (→ then ↓) to alphanumeric well IDs."""

_MF_WELL_NUMF_TO_ID = {
    (rownumber + colnumber * 4): f"{row}{column:02d}"
    for rownumber, row in enumerate("CDEF")
    for colnumber, column in enumerate(range(1, 9))
}
"""Maps 0-based well numbers in MF mode by **Fortran-style** counting order (↓ then →) to alphanumeric well IDs."""

_MF_WELL_NUMM_TO_ID = {
    1 + (colnumber + rownumber * 8): f"{row}{column:02d}"
    for rownumber, row in enumerate("CDEF")
    for colnumber, column in enumerate(reversed(range(1, 9)) if rownumber % 2 else range(1, 9))
}
"""Maps 1-based well numbers in MF mode by **measurement counting order** (→→→ ↓ ←←← ↓ →→→ ...) to alphanumeric well IDs."""

_WELL_NUMM_TO_ID = {
    1 + (colnumber + rownumber * 8): f"{row}{column:02d}"
    for rownumber, row in enumerate("ABCDEF")
    for colnumber, column in enumerate(reversed(range(1, 9)) if rownumber % 2 else range(1, 9))
}
"""Maps 1-based well numbers in non-MF mode by **measurement counting order** (→→→ ↓ ←←← ↓ →→→ ...) to alphanumeric well IDs."""


class BioLectorProParser(BLDParser):
    def parse(
        self,
        filepath: Union[str, os.PathLike],
        lot_number: Optional[int] = None,
        temp: Optional[int] = None,
        cal_0: Optional[float] = None,
        cal_100: Optional[float] = None,
        phi_min: Optional[float] = None,
        phi_max: Optional[float] = None,
        pH_0: Optional[float] = None,
        dpH: Optional[float] = None,
    ) -> BLData:
        metadata, data = parse_metadata_data(filepath)

        bld = BLData(
            model=BioLectorModel.BLPro,
            environment=extract_environment(data, metadata),
            filtersets=extract_filtersets(metadata),
            references=extract_references(data),
            measurements=extract_measurements(data),
            comments=extract_comments(data),
        )

        bld.metadata = metadata
        bld.fluidics = extract_fluidics(
            data,
            mf_mode="MF32" in metadata["main"]["mtp"],
        )
        bld.valves, bld.module = extract_valves_module(data)
        bld.diagnostics = extract_diagnostics(data)

        if lot_number is not None and temp is not None:
            lot_cal_data = fetch_calibration_data(lot_number, temp)
        else:
            lot_cal_data = None

        if lot_cal_data or (not None in [cal_0, cal_100, phi_min, phi_max, pH_0, dpH]):
            for key, fts in transform_into_filtertimeseries(
                bld.metadata, bld.measurements, bld.filtersets, True
            ):
                if (key == "pH") and (not None in [phi_min, phi_max, pH_0, dpH]):
                    fts.value = calibrate_pH(fts.value, phi_min, phi_max, pH_0, dpH)
                    bld[key] = fts
                elif (key == "pH") and lot_cal_data:
                    fts.value = calibrate_pH(
                        fts.value,
                        lot_cal_data["phi_min"],
                        lot_cal_data["phi_max"],
                        lot_cal_data["pH_0"],
                        lot_cal_data["dpH"],
                    )
                    bld[key] = fts
                elif (key == "DO") and (not None in [phi_min, phi_max, pH_0, dpH]):
                    fts.value = calibrate_DO(fts.value, cal_0, cal_100)
                    bld[key] = fts
                elif (key == "DO") and lot_cal_data:
                    fts.value = calibrate_DO(fts.value, lot_cal_data["cal_0"], lot_cal_data["cal_100"])
                    bld[key] = fts
                else:
                    bld[key] = fts
        else:
            for key, fts in transform_into_filtertimeseries(
                bld.metadata, bld.measurements, bld.filtersets, False
            ):
                bld[key] = fts

        return bld


def _filter_datalines(datalines: list) -> list:
    """Filters out unnecessary intermediate header sections."""
    datalines = [dl for l, dl in enumerate(datalines) if len(dl) > 1 and dl[1] == ";"]
    return datalines


def _parse_datalines(datalines) -> pandas.DataFrame:
    dfraw = pandas.read_csv(
        io.StringIO("".join(datalines)), sep=";", low_memory=False, converters={"Filterset": str}
    )
    return dfraw


def parse_metadata_data(fp) -> Tuple[Dict[str, Any], pandas.DataFrame]:
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata: DefaultDict[str, Any] = collections.defaultdict(dict)
    section = None
    data_start: Optional[int] = None

    for l, line in enumerate(lines):
        if line.startswith("="):
            # any section header encountered
            section = line.strip().strip("=").strip()
            if not data_start and section == "data":
                data_start = l + 1
        elif section is None:
            raise IncompatibleFileError("No metadata section header before first setting.")
        elif line.startswith("["):
            # register the value
            key, value = line.split("]")
            key = key.strip("[")
            metadata[section][key] = value.strip()
    if data_start is None:
        raise IncompatibleFileError("Section header 'data' not found.")

    # standardize the metadata keys
    metadata["date_start"] = datetime.datetime.strptime(
        metadata["process"]["start_date_time"], "%Y-%m-%d, %H:%M:%S"
    )
    if "end_process" in metadata and "end_date_time" in metadata["end_process"]:
        metadata["date_end"] = datetime.datetime.strptime(
            metadata["end_process"]["end_date_time"], "%Y-%m-%d, %H:%M:%S"
        )
    else:
        metadata["date_end"] = None

    datalines = lines[data_start:]

    datalines = _filter_datalines(datalines)
    # insert full-length section headers
    # append a ; to the header line because the P lines contain a trailing ;
    datalines.insert(
        0,
        "Type;Cycle;Well;Filterset;Time;Amp_1;Amp_2;AmpRef_1;AmpRef_2;Phase;Cal;"
        "Temp_up;Temp_down;Temp_water;O2;CO2;Humidity;Shaker;Service;User_Comment;Sys_Comment;"
        "Reservoir;MF_Volume;Temp_Ch4;T_144;T_180;T_181_1;T_192;P_Ch1;P_Ch2;P_Ch3;T_Hum;T_CO2;"
        "X-Pos;Y-Pos;T_LED;Ref_Int;Ref_Phase;Ref_Gain;Ch1-MP;Ch2-MF;Ch3-FA;Ch4-OP;Ch5-FB;IGNORE\n",
    )

    # parse the data as a DataFrame
    try:
        dfraw = _parse_datalines(datalines)
    except pandas.errors.ParserError:
        n_allowed = datalines[0].count(";")
        filtered_datalines = []
        defect_lines = []
        for l, line in enumerate(datalines):
            # ignore lines with too many columns, DEL or NUL characters
            if (
                line.count(";") <= n_allowed
                and not ("\x10" in line or "\x00" in line)
                and not re.search(r"\d*\.\d*\.\d*", line)
            ):
                filtered_datalines.append(line)
            else:
                defect_lines.append(data_start + l + 1)
        dfraw = _parse_datalines(filtered_datalines).drop_duplicates(keep="first")
        logger.warning(
            f"{fp} contains defects in lines {defect_lines}. Be extra skeptical about the parsed results."
        )

    return dict(metadata), dfraw[list(dfraw.columns)[:-1]]


def standardize(df):
    if "time" in df.columns:
        df["time"] = df["time"] / 3600
    return df


def extract_filtersets(metadata):

    # filterset-related metadata is spread over: channels, measurement channels, process
    channels = metadata.pop("channels")
    measurement_channels = metadata.pop("measurement channels")
    process = metadata["process"]

    # dictionary that will become the DataFrame
    filtersets = {fnum: {} for fnum in range(1, int(channels["no_filterset"]) + 1)}

    # grab data from measurement_channels
    for k, v in measurement_channels.items():
        num = int(k[0:2])
        key = k[3:]
        filtersets[num][key] = v

    # grab data from metadata['process']
    fname_lookup = {
        "201": "Biomass",
        "202": "pH(HP8)",
        "203": "DO(PSt3)",
        "401": "Biomass",
    }
    for fnum, fset in filtersets.items():
        for k in ["reference_value", "reference_gain", "gain"]:
            fn = fname_lookup.get(fset["filter_id"], fset["name"])
            pk = f"{fnum:02d}_{k}_{fn}"
            if pk in process:
                filtersets[fnum][k] = process.pop(pk)

    # convert to DataFrame and align column names & types
    df = pandas.DataFrame(filtersets).T
    ocol_ncol_type = [
        ("no", "filter_number", int),
        ("name", "filter_name", str),
        ("filter_id", "filter_id", str),
        ("filter_type", "filter_type", str),
        (None, "excitation", float),
        (None, "emission", float),
        # (None, 'layout', str),
        ("gain", "gain", float),
        ("gain_1", "gain_1", float),
        ("gain_2", "gain_2", float),
        (None, "phase_statistic_sigma", float),
        (None, "signal_quality_tolerance", float),
        ("reference_gain", "reference_gain", float),
        ("reference_value", "reference_value", float),
        ("calibration", "calibration", str),
        (None, "emission2", float),
    ]
    return utils.__to_typed_cols__(df, ocol_ncol_type)


def extract_comments(dfraw):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Time", "time", float),
        ("User_Comment", "user_comment", str),
        ("Sys_Comment", "sys_comment", str),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw["Type"] == "C"], ocol_ncol_type)
    return standardize(df)


def extract_parameters(dfraw):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Time", "time", float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw["Type"] == "P"], ocol_ncol_type)
    return standardize(df)


def extract_references(dfraw):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Filterset", "filterset", int),
        ("Time", "time", float),
        ("Amp_1", "amp_1", float),
        ("Amp_2", "amp_2", float),
        ("Phase", "phase", float),
    ]
    df = dfraw[dfraw["Type"] == "R"].copy()
    df["Phase"] = pandas.to_numeric(df["Phase"], errors="coerce")
    df = utils.__to_typed_cols__(df, ocol_ncol_type)
    return standardize(df).set_index(["cycle", "filterset"])


def extract_measurements(dfraw):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Well", "well", int),
        ("Filterset", "filterset", str),
        ("Time", "time", float),
        ("Amp_1", "amp_1", float),
        ("Amp_2", "amp_2", float),
        ("AmpRef_1", "amp_ref_1", float),
        ("AmpRef_2", "amp_ref_2", float),
        ("Phase", "phase", float),
        ("Cal", "cal", float),
    ]
    df_M = dfraw[dfraw["Type"] == "M"]

    # Drop lines with invalid readings
    mask = (df_M["AmpRef_1"] == "REFOVERLD") | (df_M["AmpRef_2"] == "REFOVERLD")
    ndrop = sum(mask)
    if ndrop:
        cdrop = df_M[mask]["Cycle"].to_list()
        warnings.warn(
            f"Dropped {ndrop} measurement rows from cycles {cdrop} because they have REFOVERLD.", UserWarning
        )
    df_M = df_M[~mask]

    # Drop filtersets with non-monotonically increasing time
    drop_idxs = []
    for idx, fsblock in df_M.groupby(["Cycle", "Filterset"]):
        t = fsblock["Time"].astype(int).to_numpy()
        if any(t[1:] < t[:-1]):
            drop_idxs.append(idx)
    ndrop = len(drop_idxs)
    if ndrop:
        for dropC, dropF in drop_idxs:
            mask = numpy.logical_and(df_M["Cycle"] == dropC, df_M["Filterset"] == dropF)
            df_M = df_M[~mask]
            warnings.warn(
                f"Dropped cycle {dropC} filterset {dropF} because of non-monotonically increasing time values.",
                UserWarning,
            )

    # Convert to the expected data types
    df = utils.__to_typed_cols__(df_M, ocol_ncol_type)
    df = df.set_index(["filterset", "cycle", "well"])
    return standardize(df)


def extract_environment(dfraw, metadata):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Time", "time", float),
        (None, "temp_setpoint", float),
        ("Temp_up", "temp_up", float),
        ("Temp_down", "temp_down", float),
        ("Temp_water", "temp_water", float),
        ("O2", "o2", float),
        ("CO2", "co2", float),
        ("Humidity", "humidity", float),
        (None, "shaker_setpoint", float),
        ("Shaker", "shaker_actual", float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw["Type"] == "R"], ocol_ncol_type)
    # Write initial setpoints (temp & shaker) into df
    df["humidity_setpoint"] = float(metadata["process"]["humidity"])
    df["O2_setpoint"] = float(metadata["process"]["o2"])
    df["CO2_setpoint"] = float(metadata["process"]["co2"])
    # TODO: parse setpoint changes (temp & shaker) from profiles in the metadata
    # TODO: clean up -9999.0 values in co2 column
    return standardize(df)


def extract_fluidics(dfraw, mf_mode: bool):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Well", "well", int),
        ("Time", "time", float),
        ("Reservoir", "reservoir", FluidicsSource),
        ("MF_Volume", "mf_volume", float),
        ("Temp_Ch4", "volume", float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw["Type"] == "F"], ocol_ncol_type)
    if mf_mode:
        df["well"] = [_MF_WELL_NUMM_TO_ID[w] for w in df["well"]]
    else:
        df["well"] = [_WELL_NUMM_TO_ID[w] for w in df["well"]]

    df = df.sort_values(["well", "cycle"]).set_index(["well"])
    return standardize(df)


def extract_valves_module(dfraw):
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Well", "valve", str),
        ("Filterset", "well", str),
        ("Time", "volume_1", str),
        ("Amp_1", "volume_2", str),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw["Type"] == "N"], ocol_ncol_type)

    # table of valve actions
    df_valves = df[df["valve"] != "Module 1 (BASE)"].copy()
    df_valves.columns = ["cycle", "valve", "well", "acid", "base"]
    df_valves["valve"] = df_valves["valve"].str.replace("Valve ", "").astype(int)
    df_valves["well"] = df_valves["well"].str.replace("Well", "").astype(int)
    # TODO: Which numbering style is this?
    df_valves["acid"] = df_valves["acid"].str.replace("Sollvolumen (Acid) ", "", regex=False).astype(float)
    df_valves["base"] = df_valves["base"].str.replace("Sollvolumen (Base) ", "", regex=False).astype(float)
    df_valves = standardize(df_valves).set_index(["well", "valve", "cycle"])

    # TODO: unknown column purpose
    df_module = df[df["valve"] == "Module 1 (BASE)"].copy()
    df_module.columns = ["cycle", "module", "valve", "well", "volume"]
    df_module["valve"] = df_module["valve"].str.replace("Valve ", "").astype(int)
    df_module["well"] = df_module["well"].str.replace("Well ", "").astype(int)
    # TODO: Which numbering style is this?
    df_module["volume"] = df_module["volume"].str.replace("Volume ", "").astype(float)
    df_module = standardize(df_module).set_index(["well", "valve", "cycle"])

    return df_valves, df_module


def extract_diagnostics(dfraw):
    dff = dfraw[dfraw["Type"] != "N"]
    ocol_ncol_type = [
        ("Cycle", "cycle", int),
        ("Time", "time", float),
        ("Temp_Ch4", "temp_ch4", float),
        ("T_144", "t_144", float),
        ("T_180", "t_180", float),
        ("T_181_1", "t_181_1", float),
        ("T_192", "t_192", float),
        ("P_Ch1", "p_ch1", float),
        ("P_Ch2", "p_ch2", float),
        ("P_Ch3", "p_ch3", float),
        ("T_Hum", "t_hum", float),
        ("T_CO2", "t_co2", float),
        ("X-Pos", "x-pos", float),
        ("Y-Pos", "y-pos", float),
        ("T_LED", "t_led", float),
        ("Ref_Int", "ref_int", float),
        ("Ref_Phase", "ref_phase", float),
        ("Ref_Gain", "ref_gain", float),
        ("Ch1-MP", "ch1_mp", float),
        ("Ch2-MF", "ch2_mf", float),
        ("Ch3-FA", "ch3_fa", float),
        ("Ch4-OP", "ch4_op", float),
        ("Ch5-FB", "ch5_fb", float),
    ]
    df = utils.__to_typed_cols__(dff, ocol_ncol_type)
    return standardize(df)


def transform_into_filtertimeseries(
    metadata: dict,
    measurements: pandas.DataFrame,
    filtersets: pandas.DataFrame,
    return_uncalibrated_optode_data: bool,
):
    no_to_id = {int(k.split("_")[0]): v for k, v in metadata["fermentation"].items() if k.endswith("_well")}
    for fs in filtersets.itertuples():
        filter_number = f"{fs.filter_number:02d}"
        key = None
        times = None
        values = None
        # test if any filterset is not available in measurements due to invalid data #issue24
        if filter_number not in measurements.index.get_level_values("filterset"):
            logger.warning(
                'Skipped channel %s with name "%s" because no valid measurements are available.',
                fs.filter_type,
                fs.filter_name,
            )
            continue

        dfm = measurements.xs(filter_number, level="filterset")
        # De-duplicate based on the index because in long-running experiments
        # the BioLector sometimes duplicates parts of the data.
        mask = dfm.index.duplicated(keep="first")
        if any(mask):
            logger.warning(
                "Duplicate filter %s measurements for (cycles, wells) %s.",
                filter_number,
                dfm[mask].index.to_list(),
            )
            dfm = dfm[~mask]

        if fs.filter_type == "Intensity" and ("Biomass" in fs.filter_name or "BS" in fs.filter_name):
            key = f"BS{int(fs.gain_1)}"
            times = dfm["time"].unstack()
            values = dfm["amp_ref_1"].unstack()
        elif fs.filter_type in {"pH", "DO"} and not return_uncalibrated_optode_data:
            key = fs.filter_type
            times = dfm["time"].unstack()
            values = dfm["cal"].unstack()
        elif fs.filter_type in {"pH", "DO"} and return_uncalibrated_optode_data:
            key = fs.filter_type
            times = dfm["time"].unstack()
            values = dfm["phase"].unstack()
        elif fs.filter_type == "Intensity":
            key = fs.filter_name
            times = dfm["time"].unstack()
            values = dfm["amp_ref_1"].unstack()
        else:
            logger.warn(
                f'Skipped {fs.filter_type} channel with name "{fs.filter_name}" because no processing routine is implemented.'
            )
            continue

        # transform into nicely formatted DataFrames for FilterTimeSeries
        times.columns = [no_to_id[c] for c in times.columns]
        times = times.reindex(sorted(times.columns), axis=1)
        times.columns.name = "well"
        values.columns = [no_to_id[c] for c in values.columns]
        values = values.reindex(sorted(values.columns), axis=1)
        values.columns.name = "well"
        fts = FilterTimeSeries(times, values)
        yield (key, fts)


def fetch_calibration_data(lot_number: int, temp: int):
    """Loads calibration data from calibration file. Also triggers file download.

    Parameters
    ----------
    lot_number : int
        Lot number to be used for calibration data lookup.
    temp : int
        Temperature to be used for calibration data lookup.

    Returns
    -------
    calibration_dict : dict
        Dictionary containing calibration data.
        Can be readily used in calibration function.
    """
    assert utils.__spec__ is not None
    assert utils.__spec__.origin is not None
    module_path = pathlib.Path(utils.__spec__.origin).parents[0]
    calibration_file = pathlib.Path(module_path, "cache", "CalibrationLot_II.xml")

    if not calibration_file.is_file():
        if not utils.download_calibration_data():
            return None

    def search_for_lot(calibration_file, lot_number):
        tree = xml.etree.ElementTree.parse(calibration_file)
        root = tree.getroot()

        element = None
        for i, e in enumerate(root[1].iter("Name")):
            if e.text == str(lot_number):
                element = root[1][i]
                break
        return element

    element = search_for_lot(calibration_file, lot_number)
    if not element:
        if not utils.download_calibration_data():
            return None
        else:
            element = search_for_lot(calibration_file, lot_number)

    if not element:
        raise InvalidLotNumberError(
            "Latest calibration information was downloaded from m2p-labs, "
            f"but the provided lot number/temperature combination (lot_number={lot_number}, temp={temp}) could not be found. "
            "Please check the parameters."
        )

    cp = dict()
    for p in ["fCal0_m", "fCal0_a", "fCal100_m", "fCal100_a"]:
        cp.update({p: float(element.find("DOTempCompensation").find(p).text)})

    for p in ["fMin_m", "fMin_a", "fMax_m", "fMax_a", "dpH_m", "dpH_a", "pH0_m", "pH0_a"]:
        cp.update({p: float(element.find("PHTempCompensation").find(p).text)})

    calibration_dict = {
        "cal_0": cp["fCal0_m"] * temp + cp["fCal0_a"],
        "cal_100": cp["fCal100_m"] * temp + cp["fCal100_a"],
        "phi_min": cp["fMin_m"] * temp + cp["fMin_a"],
        "phi_max": cp["fMax_m"] * temp + cp["fMax_a"],
        "pH_0": cp["pH0_m"] * temp + cp["pH0_a"],
        "dpH": cp["dpH_m"] * temp + cp["dpH_a"],
    }

    return calibration_dict


def calibrate_pH(raw, phi_min, phi_max, pH_0, dpH):
    """
    Calculation of pH:
    pH_0 + dpH * log((phi_min - phase_shift) / (phase_shift - phi_max))
    """
    kappa = raw - phi_max
    kappa[kappa <= 0] = numpy.nan

    pH = pH_0 + dpH * numpy.log((phi_min - raw) / kappa)
    return pH


def calibrate_DO(raw, cal_0, cal_100):
    """
    Calculation of DO:
    S_cal_0 = tan(cal_0 * pi / 180)
    S_cal_100 = tan(cal_100 * pi / 180)
    ksv = 0.01 * ((S_cal_0 / S_cal_100) - 1)
    DO = (1 / ksv) * ((S_cal_0 / tan(phase_shift * pi / 180)) - 1)
    """
    S_cal_0 = numpy.tan(cal_0 * numpy.pi / 180)
    S_cal_100 = numpy.tan(cal_100 * numpy.pi / 180)
    ksv = 0.01 * ((S_cal_0 / S_cal_100) - 1)
    DO = (1 / ksv) * ((S_cal_0 / numpy.tan(raw * numpy.pi / 180)) - 1)
    return DO
