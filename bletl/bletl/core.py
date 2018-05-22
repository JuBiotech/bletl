"""Specifies the base types for parsing and representing BioLector CSV files."""
import abc
import pathlib
import enum


class BLData(object):
    """Standardized data type for BioLector data.
    """
    def __init__(self):
        return super().__init__()


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