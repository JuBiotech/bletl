"""Contains unit tests for the `bletl` package"""
import numpy
import pathlib
import pandas
import unittest

import bletl
from bletl import core
from bletl import parsing


BL1_files =  [
    pathlib.Path('data', 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv'),
    pathlib.Path('data', 'BL1', 'JH_ShakerSteps_20170302_070206.csv'),
    pathlib.Path('data', 'BL1', 'NT_1400_BS20BS10_15min_20160222_151645.csv'),
    pathlib.Path('data', 'BL1', 'rj-cg-res_20170927_084112.csv'),
    pathlib.Path('data', 'BL1', 'incremental', 'NT_1400rpm_30C_BS15_5min_20180503_132133.csv'),
]
BLPro_files = list(pathlib.Path('data', 'BLPro').iterdir())
not_a_bl_file = pathlib.Path('data', 'BL1', 'incremental', 'C42.tmp')

calibration_test_file = pathlib.Path('data', 'BL1', 'NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv')
calibration_test_cal_data = {
        'cal_0': 65.91,
        'cal_100': 40.60,
        'phi_min': 57.45,
        'phi_max': 18.99,
        'pH_0': 6.46,
        'dpH': 0.56,
        }
calibration_test_times = {'BS10': (52, 'D03', 10.5221)}
calibration_test_values = {'BS10': (5, 'A04', 11.7175),
                                 'DO': (13, 'A05', 99.4285),
                                 'pH': (39, 'D08', 7.06787),
                                 'GFP75': (81, 'F07', 216.99),
                                 }

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
            with open(fp, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            headerlines, data = parsing.bl1.split_header_data(fp)

            self.assertEqual(len(headerlines) + len(data), len(lines))
        return

    def test_parsing(self):
        for fp in BL1_files:
            data = bletl.parse(fp)

            self.assertIsInstance(data.metadata, dict)
            self.assertIsInstance(data.environment, pandas.DataFrame)
            self.assertIsInstance(data.comments, pandas.DataFrame)
            self.assertIsInstance(data.measurements, pandas.DataFrame)
            self.assertIsInstance(data.references, pandas.DataFrame)
        return

    def test_temp_setpoint_parsing(self):
        fp = pathlib.Path('data', 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv')
        data = bletl.parse(fp)
        df = data.environment
        temps = set(df['temp_setpoint'].unique())
        self.assertSetEqual(temps, set([30., 25., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.]))
        # test some individual values
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 0.00778), 'temp_setpoint'])[0], 30.)
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 55.30764), 'temp_setpoint'])[0], 37.)
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 55.44867), 'temp_setpoint'])[0], 38.)
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 126.74621), 'temp_setpoint'])[0], 40.)
        return

    def test_shaker_setpoint_parsing(self):
        fp = pathlib.Path('data', 'BL1', 'JH_ShakerSteps_20170302_070206.csv')
        data = bletl.parse(fp)
        df = data.environment
        rpms = set(df['shaker_setpoint'].unique())
        self.assertSetEqual(rpms, set([
           500.,  600.,  700.,  800.,  900., 1000.,
           1100., 1200., 1300., 1400., 1500.
        ]))
        # test some individual values
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 0.06265), 'shaker_setpoint'])[0], 500.)
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 2.49103), 'shaker_setpoint'])[0], 900.)
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 2.50949), 'shaker_setpoint'])[0], 1000.)
        self.assertEqual(list(df.loc[numpy.isclose(df['time'], 5.65632), 'shaker_setpoint'])[0], 500.)
        return


class TestBL1Calibration(unittest.TestCase):
    def test_calibration_data_type(self):
        parsed_data = bletl.parse(calibration_test_file)
        parsed_data.calibrate(calibration_test_cal_data)
        data = parsed_data.calibrated_data

        for key, item in data.items():
            self.assertIsInstance(item, core.FilterTimeSeries)

    def test_calibration(self):
        parsed_data = bletl.parse(calibration_test_file)
        parsed_data.calibrate(calibration_test_cal_data)
        data = parsed_data.calibrated_data

        for key, (cycle, well, value) in calibration_test_times.items():
            self.assertAlmostEqual(data[key].time.loc[cycle, well], value, places=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            self.assertAlmostEqual(data[key].value.loc[cycle, well], value, places=4)


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
