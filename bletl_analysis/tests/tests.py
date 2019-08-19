
import numpy
import unittest
import pathlib

import bletl
import bletl_pro
import bletl_analysis
from scipy.interpolate import UnivariateSpline

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'data')
FP_TESTFILE = pathlib.Path(dir_testfiles, '107-AR_Coryne-AR-2019-04-15-12-39-30.csv')


class TestDOPeakDetection(unittest.TestCase):
    def test_find_peak(self):
        bldata = bletl.parse(FP_TESTFILE)

        x, y = bldata['DO'].get_timeseries('A01')

        c_peak = bletl_analysis.find_do_peak(x, y, delay_a=0.5, threshold_a=70, delay_b=0, threshold_b=80, initial_delay=1)

        self.assertEqual(c_peak, 60)
        return


class TestSplineMue(unittest.TestCase):
    def test_get_single_spline(self):
        bldata = bletl.parse(FP_TESTFILE)
        numpy.random.seed(9001)

        x, y = bldata['BS3'].get_timeseries('F05')

        spline = bletl_analysis._get_single_spline(x, y)
        
        self.assertIsInstance(spline, UnivariateSpline)
        self.assertAlmostEqual(spline(10), 12.4433285)
        return

    def test_get_mue_wells(self):
        bldata = bletl.parse(FP_TESTFILE)
        numpy.random.seed(9001)
        wells = 'A01,A02,B03,C05'.split(',')

        # automatic blank
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], wells=wells)
        self.assertAlmostEqual(mue_blank_first.value.loc[25, 'B03'], 0.599476, places=5)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, wells=wells)
        self.assertAlmostEqual(mue_blank_scalar.value.loc[25, 'B03'], 0.052083, places=5)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
            if well in wells
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=blank_dict)
        self.assertAlmostEqual(mue_blank_dict.value.loc[25, 'B03'], 0.534138, places=5)

        # check value error when the wells dictionary is incorrect
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=dict(A01=3, C02=4))
        # check value error on invalid blank option
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank='last')
        return

    def test_get_mue_on_all(self):
        bldata = bletl.parse(FP_TESTFILE)
        wells = list('A01,A02,B03,C05'.split(','))
        bldata['BS3'].time = bldata['BS3'].time[wells]
        bldata['BS3'].value = bldata['BS3'].value[wells]
        numpy.random.seed(9001)
        
        # automatic
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'])
        self.assertAlmostEqual(mue_blank_first.value.loc[25, 'B03'], 0.599476, places=5)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2)
        self.assertAlmostEqual(mue_blank_scalar.value.loc[25, 'B03'], 0.052083, places=5)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], blank=blank_dict)
        self.assertAlmostEqual(mue_blank_dict.value.loc[25, 'B03'], 0.534138, places=5)
        return

        
if __name__ == '__main__':
    unittest.main()
