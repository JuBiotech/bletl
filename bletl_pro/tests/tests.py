"""Contains unit tests for the `bletl_pro` package"""
import datetime
import numpy
import pathlib
import pandas
import unittest

import bletl
from bletl import core
from bletl import utils
from bletl_pro import parsing

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'data')

BL1_file = pathlib.Path(dir_testfiles, 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv')
BL2_files = list(pathlib.Path(dir_testfiles, 'BLII').iterdir())
BLPro_files = list(pathlib.Path(dir_testfiles, 'BLPro').iterdir())
calibration_test_file = pathlib.Path(dir_testfiles, 'BLPro', '18-FZJ-Test2--2018-02-07-10-01-11.csv')
incompatible_file = pathlib.Path(dir_testfiles, 'incompatible_files', 'BL2-file-saved-with-biolection.csv')

class TestParserSelection(unittest.TestCase):
    def test_selects_parsers_pro(self):
        for fp in BLPro_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.blpro.BioLectorProParser)
        return

    def test_selects_parsers_ii(self):
        for fp in BL2_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.blpro.BioLectorProParser)
        return

    def test_incompatible_file_detecion(self):
        with self.assertRaises(bletl.IncompatibleFileError):
            bletl.get_parser(incompatible_file)

    
class TestBL2Parsing(unittest.TestCase):
    def test_parse_metadata_data(self):
        for fp in BL2_files:
            metadata, data = parsing.blpro.parse_metadata_data(fp)
            
            self.assertIsInstance(metadata, dict)
            self.assertIsInstance(data, pandas.DataFrame)
        pass

    def test_parsing(self):
        for fp in BL2_files:
            try:
                data = bletl.parse(fp)
                self.assertIsInstance(data, dict)
            except:
                print('parsing failed for: {}'.format(fp))
                raise
        pass


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
                self.assertIsInstance(data, dict)
            except:
                print('parsing failed for: {}'.format(fp))
                raise
        return

    def test_parse_with_concat(self):
        data = bletl.parse(filepaths=[
                pathlib.Path(dir_testfiles, 'BLPro', '224-MO_Coryne--2019-07-12-16-54-30.csv'),
                pathlib.Path(dir_testfiles, 'BLPro', '226-MO_Coryne--2019-07-12-17-38-02.csv'),
        ])
        self.assertIsInstance(data, bletl.BLData)
        self.assertEqual(data.metadata['date_start'], datetime.datetime(2019, 7, 12, 16, 54, 30))
        self.assertEqual(data.metadata['date_end'], None)
        numpy.testing.assert_array_equal(data['BS5'].time.index, numpy.arange(1, 4+254+1))
        numpy.testing.assert_array_almost_equal(data['BS5'].time['A01'][:5], [0.013056, 0.179444, 0.346111, 0.512778, 0.738611])
        return

    def test_parse_file_with_defects(self):
        # this file has some broken & duplicate data line lines 25857-25877
        bletl.parse(pathlib.Path(dir_testfiles, 'BLPro', '247-AH_Bacillus_Batch-AH-2020-01-22-12-48-45.csv'))
        pass


class TestBLProCalibration(unittest.TestCase):
    def test_filtertimeseries_presence(self):
        bd = bletl.parse(calibration_test_file)
        self.assertIsInstance(bd, dict)
        self.assertTrue('BS2' in bd)
        self.assertTrue('BS5' in bd)
        self.assertTrue('pH' in bd)
        self.assertTrue('DO' in bd)
        return

    def test_correct_well_association(self):
        bd = bletl.parse(calibration_test_file)
        
        # F01
        x, y = bd['pH'].get_timeseries('F01')
        self.assertTrue(numpy.allclose(x[:3], [0.07972222,  0.16166667,  0.24472222]))
        self.assertTrue(numpy.allclose(y[:3], [8.15, 7.93, 7.78]))

        # D02
        x, y = bd['DO'].get_timeseries('D02')
        self.assertTrue(numpy.allclose(x[:3], [0.09166667,  0.17305556,  0.25611111]))
        self.assertTrue(numpy.allclose(y[:3], [92.09, 93.84, 94.65]))
        return


class TestDataEquivalence(unittest.TestCase):
    def test_model(self):
        data = bletl.parse(BLPro_files[0])
        
        self.assertIsInstance(data.model, core.BioLectorModel)
        self.assertEqual(data.model, core.BioLectorModel.BLPro)
        return

    def test_environment(self):
        d_1 = bletl.parse(BL1_file, 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.environment.columns), list(d_p.environment.columns))
        return

    def test_filtersets(self):
        d_1 = bletl.parse(BL1_file, 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.filtersets.columns), list(d_p.filtersets.columns))
        return

    def test_references(self):
        d_1 = bletl.parse(BL1_file, 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.references.columns), list(d_p.references.columns))
        return

    def test_measurements(self):
        d_1 = bletl.parse(BL1_file, 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.measurements.columns), list(d_p.measurements.columns))
        return

    def test_comments(self):
        d_1 = bletl.parse(BL1_file, 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.comments.columns), list(d_p.comments.columns))
        return


if __name__ == '__main__':
    unittest.main()
