"""Contains unit tests for the `bletl` package"""
import pathlib
import pandas
import unittest

import bletl
from bletl import core
from bletl import parsing


BL1_files =  [
    pathlib.Path('data', 'BL1', 'NT_1400_BS20BS10_15min_20160222_151645.csv'),
    pathlib.Path('data', 'BL1', 'rj-cg-res_20170927_084112.csv'),
    pathlib.Path('data', 'BL1', 'incremental', 'NT_1400rpm_30C_BS15_5min_20180503_132133.csv'),
]
BLPro_files = list(pathlib.Path('data', 'BLPro').iterdir())
not_a_bl_file = pathlib.Path('data', 'BL1', 'incremental', 'C42.tmp')


class TestParserSelection(unittest.TestCase):
    def test_selects_parsers(self):

        for fp in BL1_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.bl1.BioLector1Parser)

        for fp in BLPro_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.blpro.BioLectorProParser)

        return

    def test_fail_on_unsupported(self):
        self.assertRaises(NotImplementedError, bletl.get_parser, not_a_bl_file)
        return


class TestBL1Parsing(unittest.TestCase):
    def test_splitting(self):
        for fp in BL1_files:
            with open(fp, 'r') as f:
                lines = f.readlines()        
            
            headerlines, data = parsing.bl1.split_header_data(fp)

            self.assertEqual(len(headerlines) + len(data), len(lines))
        return

    def test_parsing(self):
        for fp in BL1_files:
            data = bletl.parse(fp)
            
            self.assertIsInstance(data.metadata, dict)
            self.assertIsInstance(data.process_parameters, dict)
            self.assertIsInstance(data.comments, pandas.DataFrame)
            self.assertIsInstance(data.measurements, pandas.DataFrame)
            self.assertIsInstance(data.references, pandas.DataFrame)

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
            data = bletl.parse(fp)
            pass
        return


