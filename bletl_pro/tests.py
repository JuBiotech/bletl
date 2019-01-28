"""Contains unit tests for the `bletl_pro` package"""
import numpy
import pathlib
import pandas
import unittest

import bletl
from bletl import core
from bletl import utils
from bletl_pro import parsing


BL1_files =  [
    pathlib.Path('data', 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv'),
    pathlib.Path('data', 'BL1', 'JH_ShakerSteps_20170302_070206.csv'),
    pathlib.Path('data', 'BL1', 'NT_1400_BS20BS10_15min_20160222_151645.csv'),
    pathlib.Path('data', 'BL1', 'rj-cg-res_20170927_084112.csv'),
    pathlib.Path('data', 'BL1', 'incremental', 'NT_1400rpm_30C_BS15_5min_20180503_132133.csv'),
]
BLPro_files = list(pathlib.Path('data', 'BLPro').iterdir())


class TestParserSelection(unittest.TestCase):
    def test_selects_parsers(self):
        for fp in BLPro_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.blpro.BioLectorProParser)
        return


class TestBLProParsing(unittest.TestCase):
    def test_parse_metadata_data(self):
        for fp in BLPro_files:
            metadata, data = parsing.blpro.parse_metadata_data(fp)

            self.assertIsInstance(metadata, dict)
            self.assertIsInstance(data, pandas.DataFrame)
        return

    def test_parsing(self):
        for fp in BLPro_files:
            try:
                data = bletl.parse(fp)
            except:
                print('parsing failed for: {}'.format(fp))
                raise
        return


class TestDataEquivalence(unittest.TestCase):
    def test_environment(self):
        d_1 = bletl.parse(BL1_files[0])
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.environment.columns), list(d_p.environment.columns))
        return

    def test_filtersets(self):
        d_1 = bletl.parse(BL1_files[0])
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.filtersets.columns), list(d_p.filtersets.columns))
        return

    def test_references(self):
        d_1 = bletl.parse(BL1_files[0])
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.references.columns), list(d_p.references.columns))
        return

    def test_measurements(self):
        d_1 = bletl.parse(BL1_files[0])
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.measurements.columns), list(d_p.measurements.columns))
        return

    def test_comments(self):
        d_1 = bletl.parse(BL1_files[0])
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.comments.columns), list(d_p.comments.columns))
        return