
import numpy
import unittest
import pathlib

import bletl
import bletl_pro
import bletl_analysis
import scipy.interpolate

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'data')
FP_TESTFILE = pathlib.Path(dir_testfiles, '107-AR_Coryne-AR-2019-04-15-12-39-30.csv')


class TestDOPeakDetection(unittest.TestCase):
    def test_find_peak(self):
        bldata = bletl.parse(FP_TESTFILE)

        x, y = bldata['DO'].get_timeseries('A01')

        c_peak = bletl_analysis.find_do_peak(x, y, delay_a=0.5, threshold_a=70, delay_b=0, threshold_b=80, initial_delay=1)

        self.assertEqual(c_peak, 60)
        return


class TestSplineMueScipy(unittest.TestCase):
    def test_get_single_spline(self):
        """Tests the interpolation of backscatters works with an absolute tolerance of <0.1."""
        bldata = bletl.parse(FP_TESTFILE)
        numpy.random.seed(9001)

        x, y = bldata['BS3'].get_timeseries('F05')

        spline = bletl_analysis.get_crossvalidate_spline(x, y, method='us')
        
        self.assertIsInstance(spline, scipy.interpolate.UnivariateSpline)
        
        # the last point should be very close
        numpy.testing.assert_allclose(spline(19.4275), 23.66, atol=0.01)

        # a range of points at the end of the curve
        numpy.testing.assert_allclose(spline([18.8275, 19.0275, 19.22777778]), [23.75, 23.67, 23.68], atol=0.1)
        return

    def test_get_mue_wells(self):
        """Tests that the growth rate at the end of the exponential phase is calculated with <0.001 absolute deviance."""
        bldata = bletl.parse(FP_TESTFILE)
        numpy.random.seed(9001)
        wells = 'A01,A02,B03,C05'.split(',')

        # automatic blank
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], wells=wells, method='us')
        numpy.testing.assert_allclose(mue_blank_first.value.loc[50, 'B03'], 0.375, atol=0.001)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, wells=wells, method='us')
        numpy.testing.assert_allclose(mue_blank_scalar.value.loc[50, 'B03'], 0.174, atol=0.001)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
            if well in wells
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=blank_dict, method='us')
        numpy.testing.assert_allclose(mue_blank_dict.value.loc[50, 'B03'], 0.369, atol=0.001)

        # check value error when the wells dictionary is incorrect
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=dict(A01=3, C02=4), method='us')
        # check value error on invalid blank option
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank='last', method='us')
        return

    def test_get_mue_on_all(self):
        """Tests that the growth rate at the end of the exponential phase is calculated with <0.001 absolute deviance."""
        bldata = bletl.parse(FP_TESTFILE)
        wells = list('A01,A02,B03,C05'.split(','))
        bldata['BS3'].time = bldata['BS3'].time[wells]
        bldata['BS3'].value = bldata['BS3'].value[wells]
        numpy.random.seed(9001)
        
        # automatic
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], method='us')
        numpy.testing.assert_allclose(mue_blank_first.value.loc[50, 'B03'], 0.375, atol=0.001)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, method='us')
        numpy.testing.assert_allclose(mue_blank_scalar.value.loc[50, 'B03'], 0.174, atol=0.001)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], blank=blank_dict, method='us')
        numpy.testing.assert_allclose(mue_blank_dict.value.loc[50, 'B03'], 0.369, atol=0.001)
        return


if __name__ == '__main__':
    unittest.main()
