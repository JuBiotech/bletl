"""Specifies the base types for parsing and representing BioLector CSV files."""
import abc
import pathlib
import enum
import pandas


class BLData(object):
    """Standardized data type for BioLector data.
    """
    def __init__(self, environment, filtersets, references, measurements, comments):
        self.environment = environment
        self.filtersets = filtersets
        self.references = references
        self.measurements = measurements
        self.comments = comments
        return super().__init__()

    @property
    def environment(self) -> str:
    	return self._environment

    @environment.setter
    def environment(self, value:str):
    	self._environment = value

    @property
    def filtersets(self) -> str:
    	return self._filtersets

    @filtersets.setter
    def filtersets(self, value:str):
    	self._filtersets = value

    @property
    def references(self) -> pandas.DataFrame:
    	return self._references

    @references.setter
    def references(self, value:pandas.DataFrame):
    	self._references = value

    @property
    def measurements(self) -> str:
    	return self._measurements

    @measurements.setter
    def measurements(self, value:str):
    	self._measurements = value

    @property
    def comments(self) -> pandas.DataFrame:
        return self._comments

    @comments.setter
    def comments(self, value:pandas.DataFrame):
        self._comments = value
    
    def calibrate(self, calibration_dict):
        raise NotImplementedError()


class FilterTimeSeries():
    """Generalizable data type for calibrated timeseries."""
    
    def __init__(self, time_df, value_df):
        self.time = time_df
        self.value = value_df


class BLDParser(object):
    """Abstract type for parsers that read BioLector CSV files."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def parse(self, filepath:pathlib.Path) -> BLData:
        """Parses the provided BioLector CSV file into a data object.

        Args:
            filepath (str or pathlib.Path): path pointing to the file of interest
        """
        raise NotImplementedError('Whoever implemented {} screwed up.'.format(self.__class__.__name__))


class BioLectorModel(enum.Enum):
    BL1 = 'bl1'
    BL2 = 'bl2'
    BLPro = 'blpro'