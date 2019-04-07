"""Specifies the base types for parsing and representing BioLector CSV files."""
import abc
import enum
import numpy
import pandas
import pathlib


class BioLectorModel(enum.Enum):
    BL1 = 'bl1'
    BL2 = 'bl2'
    BLPro = 'blpro'


class BLData(dict):
    """Standardized data type for BioLector data.
    """
    def __init__(self, model, environment, filtersets, references, measurements, comments):
        self._model = model
        self.environment = environment
        self.filtersets = filtersets
        self.references = references
        self.measurements = measurements
        self.comments = comments
        super().__init__()
        return 

    @property
    def model(self) -> BioLectorModel:
    	return self._model
    
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

    def get_timeseries(self, well:str) -> tuple:
        """Retrieves (time, value) for a specific well.
        
        Args:
            well (str): Well id to retrieve

        Returns:
            x (numpy.ndarray): timepoints of measurements
            y (numpy.ndarray): measured values
        """
        x = numpy.array(self.time[well])
        y = numpy.array(self.value[well])
        return x, y

    def get_unified_dataframe(self, well:str=None) -> pandas.DataFrame:
        """Retrieves a DataFrame with unified time on index.

        Args:
            well (str, optional): Well id from which time is taken. If None, the first well is used.

        Returns:
            unified_df (pd.DataFrame): Dataframe with unified time on index.
        """
        if not well is None:
            if not well in self.time.columns:
                raise ValueError('Could not find well id')
            time = self.time.loc[:, well]
        else:
            time = self.time.iloc[:, 0]

        new_index = pandas.Index(time, name='time in h')
        unified_df = self.value.set_index(new_index)
        return unified_df


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


class LotInformationError(Exception):
    pass

class InvalidLotNumberError(Exception):
    pass

class LotInformationMismatch(Warning):
    pass

class LotInformationNotFound(Warning):
    pass
