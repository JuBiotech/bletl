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

    def get_narrow_data(self) -> pandas.DataFrame:
        """Retrieves data in a narrow format. 
        
        Returns:
            narrow (pandas.DataFrame): data in a narrow format.
        """
        narrow = pandas.DataFrame(columns=['well', 'filterset', 'time', 'value'])

        for filterset, filtertimeseries in self.items():
            to_add = pandas.melt(filtertimeseries.time, value_name='time')
            to_add['value'] = pandas.melt(filtertimeseries.value, value_name='value').loc[:, 'value']
            to_add['filterset'] = filterset
            to_add.astype({'value': float})
            narrow = narrow.append(to_add, sort=False)     

        return narrow.reset_index()

    def get_unified_narrow_data(self, source_well='first', source_filterset='first') -> pandas.DataFrame:
        """Retrieves data with unified time in a narrow format. Each filterset forms a seperate column.
        
        Returns:
            u_narrow (pandas.DataFrame): data with unified time in a narrow format.

        Raises:
            KeyError: If specified source filterset or well cannot be found.
        """        
        if source_filterset == 'first':
            _source_filterset = list(self.keys())[0]
        else:
            if not source_filterset in self.keys():
                raise KeyError(f'Specified source filterset "{source_filterset}" not found.') 
            _source_filterset = source_filterset
        
        if source_well == 'first':
            _source_well = self[_source_filterset].time.columns[0]
        else:
            if not source_well in self[_source_filterset].time.columns:
                raise KeyError(f'Specified source well "{source_well}" not found.') 
            _source_well = source_well
        
        u_narrow = pandas.DataFrame(
            columns=['well', 'cycle', 'time'] + list(self.keys()),
        )

        wells = self[_source_filterset].time.columns
        cycles = self[_source_filterset].time.index
        times = self[_source_filterset].time.loc[:, _source_well].astype(float)

        u_narrow['well'] = [well for well in wells for _ in cycles]
        u_narrow['cycle'] = [cycle for _ in wells for cycle in cycles]
        u_narrow['time'] = [time for _ in wells for time in times]
        
        u_narrow = u_narrow.set_index(['well', 'cycle'])

        for filterset, filtertimeseries in self.items():
            fcycles = filtertimeseries.value.index
            fwells = filtertimeseries.value.columns
            
            molten_values = filtertimeseries.value.melt(value_name=filterset)
            molten_values['cycle'] = [cycle for _ in fwells for cycle in fcycles]
            molten_values = molten_values.set_index(['well', 'cycle'])
            u_narrow.update(molten_values)

        u_narrow = u_narrow.astype(dict(zip(self.keys(), [float]*len(self.keys()))))
        u_narrow = u_narrow.reset_index()

        return u_narrow


class FilterTimeSeries():
    """Generalizable data type for calibrated timeseries."""

    def __init__(self, time_df, value_df):
        self.time = time_df
        self.value = value_df

    def get_timeseries(self, well:str, *, last_cycle:int=None) -> tuple:
        """Retrieves (time, value) for a specific well.
        
        Args:
            well (str): Well id to retrieve
            last_cycle (int): cycle number of the last cycle to be included (defaults to all cycles)

        Returns:
            x (numpy.ndarray): timepoints of measurements
            y (numpy.ndarray): measured values
        """
        if last_cycle is not None and last_cycle <= 0:
            raise ValueError(f'last_cycle must be > 0')
        x = numpy.array(self.time[well])[:last_cycle]
        y = numpy.array(self.value[well])[:last_cycle]
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
                raise KeyError('Could not find well id')
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


class IncompatibleFileError(Exception):
    pass
