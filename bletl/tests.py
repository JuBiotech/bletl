"""Contains unit tests for the `bletl` package"""
import numpy
import pathlib
import pandas
import unittest

import bletl
from bletl import core
from bletl import parsing
from bletl import utils


BL1_files =  [
    pathlib.Path('data', 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv'),
    pathlib.Path('data', 'BL1', 'JH_ShakerSteps_20170302_070206.csv'),
    pathlib.Path('data', 'BL1', 'rj-cg-res_20170927_084112.csv'),
    pathlib.Path('data', 'BL1', 'incremental', 'NT_1400rpm_30C_BS15_5min_20180503_132133.csv'),
]

BL1_fragment_files = [
    pathlib.Path('data', 'BL1', 'fragments', 'fragment0.csv'),
    pathlib.Path('data', 'BL1', 'fragments', 'fragment1.csv'),
    pathlib.Path('data', 'BL1', 'fragments', 'fragment2.csv'),
]

BL1_files_without_calibration_info = [
    pathlib.Path('data', 'BL1', 'NT_1400_BS20BS10_15min_20160222_151645.csv'),
]

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
calibration_test_values = {
    'BS10': (5, 'A04', 11.7175),
    'DO': (13, 'A05', 99.4285),
    'pH': (39, 'D08', 7.06787),
    'GFP75': (81, 'F07', 216.99),
}
calibration_test_lot_number = 1515
calibration_test_temp = 30

file_with_lot_info = pathlib.Path('data', 'BL1', 'example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv')

class TestParserSelection(unittest.TestCase):
    def test_selects_parsers(self):

        for fp in BL1_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.bl1.BioLector1Parser)

        return

    def test_fail_on_unsupported(self):
        self.assertRaises(NotImplementedError, bletl.get_parser, not_a_bl_file)
        return


class TestUtils(unittest.TestCase):
    def test_last_well_in_cycle(self):
        measurements = pandas.DataFrame(data={
            'cycle': [1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
            'filterset': [1,1,1,1,2,2,2,2,3,3,3,3] * 2,
            'well': ['A01', 'A02', 'A03', 'A04'] * 3 * 2
        })
        last_well = utils._last_well_in_cycle(measurements)
        self.assertEqual(last_well, 'A04')

        # cut off the last two measurements
        measurements = measurements.iloc[:-2]
        return

    def test_last_full_cycle(self):
        measurements = pandas.DataFrame(data={
            'cycle': [1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
            'filterset': [1,1,1,1,2,2,2,2,3,3,3,3] * 2,
            'well': ['A01', 'A02', 'A03', 'A04'] * 3 * 2
        })

        last_cycle = utils._last_full_cycle(measurements)
        self.assertEqual(last_cycle, 2)

        # cut off the last two measurements
        measurements = measurements.iloc[:-2]

        last_cycle = utils._last_full_cycle(measurements)
        self.assertEqual(last_cycle, 1)
        return

    def test_cal_info_parsing(self):
        example = '1724-hc-Temp37'
        lot_number, temp = utils._parse_calibration_info(example)
        self.assertEqual(lot_number, 1724)
        self.assertEqual(temp, 37)
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

            self.assertIsInstance(data.model, core.BioLectorModel)
            self.assertEqual(data.model, core.BioLectorModel.BL1)
            self.assertIsInstance(data.metadata, dict)
            self.assertIsInstance(data.environment, pandas.DataFrame)
            self.assertIsInstance(data.comments, pandas.DataFrame)
            self.assertIsInstance(data.measurements, pandas.DataFrame)
            self.assertIsInstance(data.references, pandas.DataFrame)
        return

    def test_concat_parsing(self):
        filepaths = BL1_fragment_files
        data = bletl.parse(filepaths)
        self.assertIsInstance(data.metadata, dict)
        self.assertIsInstance(data.environment, pandas.DataFrame)
        self.assertIsInstance(data.comments, pandas.DataFrame)
        self.assertIsInstance(data.measurements, pandas.DataFrame)
        self.assertIsInstance(data.references, pandas.DataFrame)
        return

    def test_incomplete_cycle_drop(self):
        filepath = BL1_files[2]
        data = bletl.parse(filepath, drop_incomplete_cycles=False)
        self.assertEqual(data.measurements.index[-1], (3, 179, 'C08'))

        data = bletl.parse(filepath, drop_incomplete_cycles=True)
        self.assertEqual(data.measurements.index[-1], (3, 178, 'F01'))
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
        parsed_data = bletl.parse(calibration_test_file, calibration_test_lot_number, calibration_test_temp)
        data = parsed_data.calibrated_data

        for _, item in data.items():
            self.assertIsInstance(item, core.FilterTimeSeries)

        return

    def test_single_file_with_lot(self):
        parsed_data = bletl.parse(calibration_test_file, calibration_test_lot_number, calibration_test_temp)
        data = parsed_data.calibrated_data

        for key, (cycle, well, value) in calibration_test_times.items():
            self.assertAlmostEqual(data[key].time.loc[cycle, well], value, places=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            self.assertAlmostEqual(data[key].value.loc[cycle, well], value, places=4)

        return

    def test_single_file_with_caldata(self):
        parsed_data = bletl.parse_with_calibration_parameters(calibration_test_file, **calibration_test_cal_data)
        data = parsed_data.calibrated_data

        for key, (cycle, well, value) in calibration_test_times.items():
            self.assertAlmostEqual(data[key].time.loc[cycle, well], value, places=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            self.assertAlmostEqual(data[key].value.loc[cycle, well], value, places=4)

        return

    def test_fragments_with_lot(self):
        filepaths = BL1_fragment_files
        parsed_data = bletl.parse(filepaths, 1846, 37)
        data = parsed_data.calibrated_data

        self.assertAlmostEqual(data['DO'].value.loc[666, 'F07'], 12.1887, places=4)
        self.assertAlmostEqual(data['pH'].value.loc[507, 'E06'], 6.6435, places=4)

        return

    def test_fragments_with_cal_data(self):
        filepaths = BL1_fragment_files
        parsed_data = bletl.parse_with_calibration_parameters(
            filepaths=filepaths,
            cal_0=71.93,
            cal_100=38.64,
            phi_min=55.36,
            phi_max=11.91,
            pH_0=6.05,
            dpH=0.53,
            drop_incomplete_cycles=True,
        )
        data = parsed_data.calibrated_data

        self.assertAlmostEqual(data['DO'].value.loc[666, 'F07'], 12.1887, places=4)
        self.assertAlmostEqual(data['pH'].value.loc[507, 'E06'], 6.6435, places=4)

        return

    def test_mismatch_warning(self):
        with self.assertRaises(core.LotInformationMismatch):
            bletl.parse(file_with_lot_info, 1818, 37)
        return


class TestOnlineMethods(unittest.TestCase):
    def test_get_calibration_dict(self):
        cal_dict_fetched = bletl.fetch_calibration_data(1515, 30)
        self.assertDictEqual(cal_dict_fetched, calibration_test_cal_data)
        return

    def test_invalid_lot_number(self):
        with self.assertRaises(core.InvalidLotNumberError):
            bletl.fetch_calibration_data(99, 99)
        return

    def test_download_calibration_data(self):
        bletl.download_calibration_data()        
        return
