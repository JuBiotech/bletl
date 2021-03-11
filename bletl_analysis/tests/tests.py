
import numpy
import unittest
import pathlib
import pandas

import bletl
import bletl_analysis
from bletl_analysis import features
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

        x, y = bldata['BS3'].get_timeseries('F05')

        spline = bletl_analysis.get_crossvalidated_spline(x, y, method='us')
        
        self.assertIsInstance(spline, scipy.interpolate.UnivariateSpline)

        # the last point should be very close
        numpy.testing.assert_allclose(spline(19.4275), 23.66, atol=0.1)

        # a range of points at the end of the curve
        numpy.testing.assert_allclose(spline([18.8275, 19.0275, 19.22777778]), [23.75, 23.67, 23.68], atol=0.1)
        return

    def test_get_mue_wells(self):
        """Tests that the median growth rate over 15 exponential cycles is calculated with <0.02 absolute deviance."""
        bldata = bletl.parse(FP_TESTFILE)
        wells = 'A01,A02,B03,C05'.split(',')
 
        # automatic blank
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], wells=wells, method='us')
        mue_median = numpy.median(mue_blank_first.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.01)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, wells=wells, method='us')
        mue_median = numpy.median(mue_blank_scalar.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.30, atol=0.01)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
            if well in wells
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=blank_dict, method='us')
        mue_median = numpy.median(mue_blank_dict.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.01)

        # check value error when the wells dictionary is incorrect
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=dict(A01=3, C02=4), method='us')
        # check value error on invalid blank option
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank='last', method='us')
        return

    def test_get_mue_on_all(self):
        """Tests that the median growth rate over 15 exponential cycles is calculated with <0.02 absolute deviance."""
        bldata = bletl.parse(FP_TESTFILE)
        wells = list('A01,A02,B03,C05'.split(','))
        bldata['BS3'].time = bldata['BS3'].time[wells]
        bldata['BS3'].value = bldata['BS3'].value[wells]
        
        # automatic
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], method='us')
        mue_median = numpy.median(mue_blank_first.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.01)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, method='us')
        mue_median = numpy.median(mue_blank_scalar.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.30, atol=0.01)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], blank=blank_dict, method='us')
        mue_median = numpy.median(mue_blank_dict.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.01)
        return


class TestSplineMueCsaps(unittest.TestCase):
    def test_get_single_spline(self):
        """Tests the interpolation of backscatters works with an absolute tolerance of <0.1."""
        bldata = bletl.parse(FP_TESTFILE)

        x, y = bldata['BS3'].get_timeseries('F05')

        spline = bletl_analysis.get_crossvalidated_spline(x, y, method='ucss')
        
        self.assertIsInstance(spline, bletl_analysis.UnivariateCubicSmoothingSpline)

        # the last point should be very close
        numpy.testing.assert_allclose(spline(19.4275), 23.66, atol=0.1)

        # a range of points at the end of the curve
        numpy.testing.assert_allclose(spline([18.8275, 19.0275, 19.22777778]), [23.75, 23.67, 23.68], atol=0.1)
        return

    def test_get_mue_wells(self):
        """Tests that the median growth rate over 15 exponential cycles is calculated with <0.02 absolute deviance."""
        bldata = bletl.parse(FP_TESTFILE)
        wells = 'A01,A02,B03,C05'.split(',')

        # automatic blank
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], wells=wells, method='ucss')
        mue_median = numpy.median(mue_blank_first.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.02)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, wells=wells, method='ucss')
        mue_median = numpy.median(mue_blank_scalar.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.30, atol=0.02)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
            if well in wells
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=blank_dict, method='ucss')
        mue_median = numpy.median(mue_blank_dict.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.02)

        # check value error when the wells dictionary is incorrect
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank=dict(A01=3, C02=4), method='ucss')
        # check value error on invalid blank option
        with self.assertRaises(ValueError):
            bletl_analysis.get_mue(bldata['BS3'], wells=wells, blank='last', method='us')
        return

    def test_get_mue_on_all(self):
        """Tests that the median growth rate over 15 exponential cycles is calculated with <0.02 absolute deviance."""
        bldata = bletl.parse(FP_TESTFILE)
        wells = list('A01,A02,B03,C05'.split(','))
        bldata['BS3'].time = bldata['BS3'].time[wells]
        bldata['BS3'].value = bldata['BS3'].value[wells]
        
        # automatic
        mue_blank_first = bletl_analysis.get_mue(bldata['BS3'], method='ucss')
        mue_median = numpy.median(mue_blank_first.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.02)

        # scalar blank for all
        mue_blank_scalar = bletl_analysis.get_mue(bldata['BS3'], blank=2, method='ucss')
        mue_median = numpy.median(mue_blank_scalar.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.30, atol=0.02)
        
        # dictionary of scalars (first 5 cycles)
        blank_dict = {
            well : data.iloc[:5].mean()
            for well, data in bldata['BS3'].value.iteritems()
        }
        mue_blank_dict = bletl_analysis.get_mue(bldata['BS3'], blank=blank_dict, method='ucss')
        mue_median = numpy.median(mue_blank_dict.value.loc[60:75, 'B03'])
        numpy.testing.assert_allclose(mue_median, 0.38, atol=0.02)
        return


class TestSplineMethodEquivalence(unittest.TestCase):
    def test_API_comparable(self):
        x = numpy.linspace(0, 50, 20)
        y = numpy.random.normal(x)
        ucss = bletl_analysis.get_crossvalidated_spline(x, y, method='ucss')
        self.assertIsInstance(ucss, bletl_analysis.UnivariateCubicSmoothingSpline)
        der_1 = ucss.derivative(1)
        der_2 = ucss.derivative(2)
        der_3 = ucss.derivative(3)
        self.assertIsInstance(der_1(x), numpy.ndarray)
        self.assertIsInstance(der_1(5.3), float)
        self.assertIsNotNone(der_1)
        self.assertIsNotNone(der_2)
        self.assertIsNotNone(der_3)
        self.assertIsNotNone(der_1(x))
        self.assertIsNotNone(der_2(x))
        self.assertIsNotNone(der_3(x))
        return

    def test_ideal_exponential_mue(self):
        # generate data of ideal exponential growth
        mue = 0.352
        y0 = 0.05
        t = numpy.linspace(0, 20, 50)
        y = y0 * numpy.exp(mue*t)

        numpy.random.seed(25)
        spline_us = bletl_analysis.get_crossvalidated_spline(t, y, method='us')
        spline_ucss = bletl_analysis.get_crossvalidated_spline(t, y, method='ucss')
        numpy.random.seed(None)

        # test that both spline approximations have residuals of less than 3 % of the signal amplitude
        diff_us = numpy.abs(spline_us(t) - y)
        diff_ucss = numpy.abs(spline_ucss(t) - y)
        self.assertTrue(numpy.all(numpy.max(diff_us) < numpy.ptp(y)*0.03))
        self.assertTrue(numpy.all(numpy.max(diff_ucss) < numpy.ptp(y)*0.03))

        # test that the median specific growth rate is close to the true value
        mue_us = spline_us.derivative(1)(t[1:]) / spline_us(t[1:])
        mue_ucss = spline_ucss.derivative(1)(t[1:]) / spline_us(t[1:])
        
        numpy.testing.assert_almost_equal(numpy.median(mue_us), mue, decimal=2)
        numpy.testing.assert_almost_equal(numpy.median(mue_ucss), mue, decimal=2)
        return


class TestFeatureExtraction(unittest.TestCase):
    def test_feature_extraction(self):
        bldata = bletl.parse(FP_TESTFILE)
        # extraction with last_cycles
        extractors = {
            "BS3": [features.TSFreshExtractor(), features.BSFeatureExtractor()],
            "DO": [features.TSFreshExtractor(), features.DOFeatureExtractor()],
            "pH": [features.TSFreshExtractor(), features.pHFeatureExtractor()],
        }
        extracted_features = features.from_bldata(
            bldata,
            extractors,
            last_cycles = {
                "A01" : 20,
                "B01" : 50,
            }
        )
        assert isinstance(extracted_features, pandas.DataFrame)
        assert extracted_features.loc["A01", "pH_x__maximum"] == bldata.get_timeseries("pH", "A01", last_cycle=20)[1].max()
        assert extracted_features.loc["B01", "pH_x__sum_values"] == bldata.get_timeseries("pH", "B01", last_cycle=50)[1].sum()
        assert extracted_features.loc["C03", "pH_x__sum_values"] == bldata.get_timeseries("pH", "C03")[1].sum()
        
        # extraction without last_cycles
        extracted_features = features.from_bldata(bldata, extractors, None)
        assert isinstance(extracted_features, pandas.DataFrame)
        
        # extraction with invalid filterset
        extractors = {
            "xyz" : [features.BSFeatureExtractor()]
        }
        with self.assertRaises(KeyError):
            features.from_bldata(bldata, extractors, None)
        return


if __name__ == '__main__':
    unittest.main()
