"""Specifies the base types for parsing and representing BioLector CSV files."""
import abc
import enum
import os
import typing
from typing import Dict, Optional, Tuple, Union

import numpy
import pandas


class BioLectorModel(enum.Enum):
    """Enumeration of BioLector Models."""

    BL1 = "bl1"
    BL2 = "bl2"
    BLPro = "blpro"
    XT = "blXT"


class FluidicsSource(enum.IntEnum):
    """Number that identifies the source of volume changes."""

    ReservoirA = 1

    """Additions from reservoir A."""
    ReservoirB = 2

    """Additions from reservoir B."""

    Pipetting = -1
    """Additions from pipetting."""


class FilterTimeSeries:
    """Generalizable data type for calibrated timeseries."""

    @property
    def wells(self) -> typing.Tuple[str, ...]:
        """Well IDs that were measured."""
        return tuple(self.time.columns)

    def __init__(self, time_df: pandas.DataFrame, value_df: pandas.DataFrame):
        self.time = time_df
        self.value = value_df

    def get_timeseries(
        self, well: str, *, last_cycle: Optional[int] = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Retrieves (time, value) for a specific well.

        Parameters
        ----------
        well : str
            Well id to retrieve.
        last_cycle : int, optional
            Cycle number of the last cycle to be included (defaults to all cycles).

        Returns
        -------
        x : numpy.ndarray
            Timepoints of measurements.
        y : numpy.ndarray
            Measured values.
        """
        if last_cycle is not None and last_cycle <= 0:
            raise ValueError(f"last_cycle must be > 0")
        x = numpy.array(self.time[well])[:last_cycle]
        y = numpy.array(self.value[well])[:last_cycle]
        return x, y

    def get_unified_dataframe(self, well: Optional[str] = None) -> pandas.DataFrame:
        """Retrieves a DataFrame with unified time on index.

        Parameters
        ----------
        well : str, optional
            Well id from which time is taken.
            If `None`, the first well is used.

        Returns
        -------
        unified_df : pandas.DataFrame
            Dataframe with unified time on index.
        """
        if not well is None:
            if not well in self.time.columns:
                raise KeyError("Could not find well id")
            time = self.time.loc[:, well]
        else:
            time = self.time.iloc[:, 0]

        new_index = pandas.Index(time, name="time in h")
        unified_df = self.value.set_index(new_index)
        return unified_df

    def __repr__(self):
        return f"FilterTimeSeries({len(self.time)} cycles, {len(self.time.columns)} wells)"


class BLData(Dict[str, FilterTimeSeries]):
    """Standardized data type for BioLector data."""

    def __init__(
        self,
        model: BioLectorModel,
        environment: pandas.DataFrame,
        filtersets: pandas.DataFrame,
        references: pandas.DataFrame,
        measurements: pandas.DataFrame,
        comments: pandas.DataFrame,
    ):
        self._model = model
        self._environment = environment
        self._filtersets = filtersets
        self._references = references
        self._measurements = measurements
        self._comments = comments
        # Optional, depending on the BioLector Model
        self.metadata: dict = {}
        self.fluidics: Optional[pandas.DataFrame] = None
        self.module: Optional[pandas.DataFrame] = None
        self.valves: Optional[pandas.DataFrame] = None
        self.diagnostics: Optional[pandas.DataFrame] = None
        super().__init__()

    @property
    def model(self) -> BioLectorModel:
        """BioLector model that the dataset was acquired with."""
        return self._model

    @property
    def environment(self) -> pandas.DataFrame:
        """Temperature, humidity etc. measurements."""
        return self._environment

    @property
    def filtersets(self) -> pandas.DataFrame:
        """Filtersets that were used in this process."""
        return self._filtersets

    @property
    def wells(self) -> typing.Tuple[str, ...]:
        """Wells that were measured."""
        if len(self) == 0:
            return tuple()
        return tuple(self.values())[0].wells

    @property
    def references(self) -> pandas.DataFrame:
        """Reference measurements that are used for calibration."""
        return self._references

    @property
    def measurements(self) -> pandas.DataFrame:
        """Well-wise filterset measurements."""
        return self._measurements

    @property
    def comments(self) -> pandas.DataFrame:
        """User and system comments."""
        return self._comments

    def get_narrow_data(self) -> pandas.DataFrame:
        """Retrieves data in a narrow format.

        Returns
        -------
        narrow : pandas.DataFrame
            Data in a narrow format.
        """
        narrow = pandas.DataFrame(columns=["well", "filterset", "time", "value"])

        for filterset, filtertimeseries in self.items():
            to_add = pandas.melt(filtertimeseries.time, value_name="time")
            to_add["value"] = pandas.melt(filtertimeseries.value, value_name="value").loc[:, "value"]
            to_add["filterset"] = filterset
            to_add.astype({"value": float})
            narrow = pandas.concat([narrow, to_add], sort=False)

        return narrow.reset_index()

    def get_unified_narrow_data(
        self,
        source_well: str = "first",
        source_filterset: str = "first",
        *,
        last_cycles: Optional[Dict[str, int]] = None,
    ) -> pandas.DataFrame:
        """Retrieves data with unified time in a narrow format. Each filterset forms a seperate column.

        Parameters
        ----------
        source_well : str
            Either "first", or the ID of the well from which timestamps are taken.
        source_filterset : str
            Either "first", or the ID of the filterset from which timestamps are taken.
        last_cycles : dict, optional
            Dictionary of well-wise maximum cycle numbers to retrieve.
            The cycle numbers in this dictionary will be included.

        Returns
        -------
        u_narrow : pandas.DataFrame
            Data with unified time in a narrow format.

        Raises
        ------
        KeyError
            If specified source filterset or well cannot be found.
        """
        if source_filterset == "first":
            _source_filterset = list(self.keys())[0]
        else:
            if not source_filterset in self.keys():
                raise KeyError(f'Specified source filterset "{source_filterset}" not found.')
            _source_filterset = source_filterset

        if source_well == "first":
            _source_well = self[_source_filterset].time.columns[0]
        else:
            if not source_well in self[_source_filterset].time.columns:
                raise KeyError(f'Specified source well "{source_well}" not found.')
            _source_well = source_well

        u_narrow = pandas.DataFrame(
            columns=["well", "cycle", "time"] + list(self.keys()),
        )

        wells = self[_source_filterset].time.columns
        cycles = self[_source_filterset].time.index
        times = self[_source_filterset].time.loc[:, _source_well].astype(float)

        u_narrow["well"] = [well for well in wells for _ in cycles]
        u_narrow["cycle"] = [cycle for _ in wells for cycle in cycles]
        u_narrow["time"] = [time for _ in wells for time in times]

        u_narrow = u_narrow.set_index(["well", "cycle"])

        for filterset, filtertimeseries in self.items():
            fcycles = filtertimeseries.value.index
            fwells = filtertimeseries.value.columns

            molten_values = filtertimeseries.value.melt(value_name=filterset)
            molten_values["cycle"] = [cycle for _ in fwells for cycle in fcycles]
            molten_values = molten_values.set_index(["well", "cycle"])
            u_narrow.update(molten_values)

        u_narrow = u_narrow.astype(dict(zip(self.keys(), [float] * len(self.keys()))))
        u_narrow = u_narrow.reset_index()

        if last_cycles:
            for well, last_cycle in last_cycles.items():
                u_narrow.drop(
                    u_narrow[(u_narrow.well == well) & (u_narrow.cycle > last_cycle)].index, inplace=True
                )

        return u_narrow

    def get_timeseries(
        self, filterset: str, well: str, *, last_cycle: Optional[int] = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Retrieves (time, value) for a specific well in a specified filterset.

        Parameters
        ----------
        filterset : str
            Name of the filterset to read from.
        well : str
            Well id to retrieve.
        last_cycle : int
            Cycle number of the last cycle to be included (defaults to all cycles).

        Returns
        -------
        x : numpy.ndarray
            Timepoints of measurements.
        y : numpy.ndarray
            Measured values.
        """
        return self[filterset].get_timeseries(well, last_cycle=last_cycle)

    def __repr__(self):
        return (
            f"BLData(model={self.model.name})"
            + " {\n"
            + "\n".join([f'  "{key}": {fts.__repr__()},' for key, fts in self.items()])
            + "\n}"
        )


class BLDParser:
    """Abstract type for parsers that read BioLector CSV files."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def parse(
        self,
        filepath: Union[str, os.PathLike],
        *,
        lot_number: Optional[int] = None,
        temp: Optional[int] = None,
        cal_0: Optional[float] = None,
        cal_100: Optional[float] = None,
        phi_min: Optional[float] = None,
        phi_max: Optional[float] = None,
        pH_0: Optional[float] = None,
        dpH: Optional[float] = None,
    ) -> BLData:
        """Parses the provided BioLector CSV file into a data object.

        If any calibration parameters are passed, all of them must be passed.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path pointing to the file of interest.
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
        """
        raise NotImplementedError(f"Whoever implemented {self.__class__.__name__} screwed up.")


class LotInformationError(Exception):
    pass


class InvalidLotNumberError(Exception):
    pass


class LotInformationMismatch(UserWarning):
    pass


class LotInformationNotFound(UserWarning):
    pass


class IncompatibleFileError(Exception):
    pass


class NoMeasurementData(UserWarning):
    pass
