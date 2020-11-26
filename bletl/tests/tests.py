"""Contains unit tests for the `bletl` package"""
import numpy
import pathlib
import pandas
import unittest
import datetime

import bletl
from bletl import core
from bletl import parsing
from bletl import utils

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'data')

BL1_files =  [
    pathlib.Path(dir_testfiles, 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv'),
    pathlib.Path(dir_testfiles, 'BL1', 'JH_ShakerSteps_20170302_070206.csv'),
    pathlib.Path(dir_testfiles, 'BL1', 'rj-cg-res_20170927_084112.csv'),
    pathlib.Path(dir_testfiles, 'BL1', 'incremental', 'NT_1400rpm_30C_BS15_5min_20180503_132133.csv'),
]

BL1_fragment_files = [
    pathlib.Path(dir_testfiles, 'BL1', 'fragments', 'fragment0.csv'),
    pathlib.Path(dir_testfiles, 'BL1', 'fragments', 'fragment1.csv'),
    pathlib.Path(dir_testfiles, 'BL1', 'fragments', 'fragment2.csv'),
]

BL1_files_without_calibration_info = [
    pathlib.Path(dir_testfiles, 'BL1', 'NT_1400_BS20BS10_15min_20160222_151645.csv'),
]

not_a_bl_file = pathlib.Path(dir_testfiles, 'BL1', 'incremental', 'C42.tmp')

BL2_files = list(pathlib.Path(dir_testfiles, 'BLII').iterdir())
BLPro_files = list(pathlib.Path(dir_testfiles, 'BLPro').iterdir())
calibration_test_file = pathlib.Path(dir_testfiles, 'BLPro', '18-FZJ-Test2--2018-02-07-10-01-11.csv')
incompatible_file = pathlib.Path(dir_testfiles, 'incompatible_files', 'BL2-file-saved-with-biolection.csv')

calibration_test_file_pro = pathlib.Path(dir_testfiles, 'BLPro', '18-FZJ-Test2--2018-02-07-10-01-11.csv')
calibration_test_file = pathlib.Path(dir_testfiles, 'BL1', 'NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv')
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

file_with_lot_info = pathlib.Path(dir_testfiles, 'BL1', 'example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv')


class TestParserSelection(unittest.TestCase):
    def test_selects_parsers(self):
        for fp in BL1_files:
            parser = bletl.get_parser(fp)
            self.assertIsInstance(parser, core.BLDParser)
            self.assertIsInstance(parser, parsing.bl1.BioLector1Parser)

        return

    def test_fail_on_unsupported(self):
        self.assertRaises(ValueError, bletl.get_parser, not_a_bl_file)
        return

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
        fp = pathlib.Path(dir_testfiles, 'BL1', 'NT_1400rpm_30C_BS15_5min_20180618_102917.csv')
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
        fp = pathlib.Path(dir_testfiles, 'BL1', 'JH_ShakerSteps_20170302_070206.csv')
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

    def test_get_timeseries(self):
        fp = pathlib.Path(dir_testfiles, 'BL1', 'NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv')
        data = bletl.parse(fp, lot_number=calibration_test_lot_number, temp=30)
        x, y = data['pH'].get_timeseries('A03')
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 103)
        self.assertEqual(numpy.sum(x), 1098.50712)
        self.assertEqual(numpy.sum(y), 746.2625060506602)
        return

    def test_get_timeseries_last_cycle(self):
        fp = pathlib.Path(dir_testfiles, 'BL1', 'NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv')
        data = bletl.parse(fp, lot_number=calibration_test_lot_number, temp=30)
        
        # default to all
        x, y = data['pH'].get_timeseries('A03')
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 103)

        # invalid values
        with self.assertRaises(ValueError):
            data['pH'].get_timeseries('A03', last_cycle=-1)

        with self.assertRaises(ValueError):
            data['pH'].get_timeseries('A03', last_cycle=0)

        # valid settings
        x, y = data['pH'].get_timeseries('A03', last_cycle=1)
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), 1)

        x, y = data['pH'].get_timeseries('A03', last_cycle=50)
        self.assertEqual(len(x), 50)
        self.assertEqual(len(y), 50)

        # more than available
        x, y = data['pH'].get_timeseries('A03', last_cycle=200)
        self.assertEqual(len(x), 103)
        self.assertEqual(len(y), 103)
        pass

    def test_get_unified_dataframe(self):
        fp = pathlib.Path(dir_testfiles, 'BL1', 'example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv')
        data = bletl.parse(fp)
        unified_df = data['BS20'].get_unified_dataframe()
        self.assertIsInstance(unified_df, pandas.DataFrame)
        self.assertAlmostEqual(unified_df.index[2], 0.35313, places=4)
        self.assertAlmostEqual(
            unified_df.iloc[
                unified_df.index.get_loc(5, method='nearest'),
                unified_df.columns.get_loc('A05')
                ],
            63.8517,
            places=4,
        )

    def test_get_narrow_data(self):
        fp = pathlib.Path(dir_testfiles, 'BL1', 'example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv')
        data = bletl.parse(fp)
        narrow_data = data.get_narrow_data()
        self.assertIsInstance(narrow_data, pandas.DataFrame)
        self.assertAlmostEqual(
            narrow_data.loc[52884, 'value'],
            100.8726865,
            places=4
        )

    def test_get_unified_narrow_data(self):
        fp = pathlib.Path(dir_testfiles, 'BL1', 'example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv')
        data = bletl.parse(fp)

        unified_narrow_data_1 = data.get_unified_narrow_data()
        self.assertIsInstance(unified_narrow_data_1, pandas.DataFrame)
        self.assertAlmostEqual(
            unified_narrow_data_1.loc[22272, 'pH'],
            7.352193857,
            places=4
        )

        unified_narrow_data_2 = data.get_unified_narrow_data(source_filterset='DO', source_well='B04')
        self.assertAlmostEqual(
            unified_narrow_data_2.loc[0, 'time'],
            0.09409,
            places=4
        )

        with self.assertRaises(KeyError):
            data.get_unified_narrow_data(source_filterset='machine_that_goes_ping')
        
        with self.assertRaises(KeyError):
            data.get_unified_narrow_data(source_well='O9000')

class TestBL1Calibration(unittest.TestCase):
    def test_calibration_data_type(self):
        data = bletl.parse(calibration_test_file, calibration_test_lot_number, calibration_test_temp)
        
        for _, item in data.items():
            self.assertIsInstance(item, core.FilterTimeSeries)

        return

    def test_single_file_with_lot(self):
        data = bletl.parse(calibration_test_file, calibration_test_lot_number, calibration_test_temp)
       
        for key, (cycle, well, value) in calibration_test_times.items():
            self.assertAlmostEqual(data[key].time.loc[cycle, well], value, places=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            self.assertAlmostEqual(data[key].value.loc[cycle, well], value, places=4)

        return

    def test_single_file_with_caldata(self):
        data = bletl.parse_with_calibration_parameters(calibration_test_file, **calibration_test_cal_data)
        
        for key, (cycle, well, value) in calibration_test_times.items():
            self.assertAlmostEqual(data[key].time.loc[cycle, well], value, places=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            self.assertAlmostEqual(data[key].value.loc[cycle, well], value, places=4)

        return

    def test_fragments_with_lot(self):
        filepaths = BL1_fragment_files
        data = bletl.parse(filepaths, 1846, 37)
        
        self.assertAlmostEqual(data['DO'].value.loc[666, 'F07'], 12.1887, places=4)
        self.assertAlmostEqual(data['pH'].value.loc[507, 'E06'], 6.6435, places=4)

        return

    def test_fragments_with_cal_data(self):
        filepaths = BL1_fragment_files
        data = bletl.parse_with_calibration_parameters(
            filepaths=filepaths,
            cal_0=71.93,
            cal_100=38.64,
            phi_min=55.36,
            phi_max=11.91,
            pH_0=6.05,
            dpH=0.53,
            drop_incomplete_cycles=True,
        )
        
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

    def test_parse_metadata_data_new_format(self):
        fp = pathlib.Path(dir_testfiles, 'BLPro', 'Ms000367.LG.csv')
        metadata, data = parsing.blpro.parse_metadata_data(fp)
        assert isinstance(metadata, dict)
        assert isinstance(data, pandas.DataFrame)
        pass

    def test_parse_new_format(self):
        fp = pathlib.Path(dir_testfiles, 'BLPro', 'Ms000367.LG.csv')
        bldata = bletl.parse(fp)
        assert 'BS3' in bldata
        t, y = bldata['BS3'].get_timeseries('A01')
        numpy.testing.assert_array_almost_equal(t, [0.01111111, 0.08888889])
        numpy.testing.assert_array_almost_equal(y, [4.19, 1.96])
        pass

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
        bd = bletl.parse(calibration_test_file_pro)
        self.assertIsInstance(bd, dict)
        self.assertTrue('BS2' in bd)
        self.assertTrue('BS5' in bd)
        self.assertTrue('pH' in bd)
        self.assertTrue('DO' in bd)
        return

    def test_correct_well_association(self):
        bd = bletl.parse(calibration_test_file_pro)
        
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
        d_1 = bletl.parse(BL1_files[0], 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.environment.columns), list(d_p.environment.columns))
        return

    def test_filtersets(self):
        d_1 = bletl.parse(BL1_files[0], 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.filtersets.columns), list(d_p.filtersets.columns))
        return

    def test_references(self):
        d_1 = bletl.parse(BL1_files[0], 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.references.columns), list(d_p.references.columns))
        return

    def test_measurements(self):
        d_1 = bletl.parse(BL1_files[0], 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.measurements.columns), list(d_p.measurements.columns))
        return

    def test_comments(self):
        d_1 = bletl.parse(BL1_files[0], 1818, 30)
        d_p = bletl.parse(BLPro_files[0])

        self.assertSequenceEqual(list(d_1.comments.columns), list(d_p.comments.columns))
        return


if __name__ == '__main__':
    unittest.main()
