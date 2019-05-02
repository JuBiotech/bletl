
import numpy
import unittest
import pathlib

import bletl
import bletl_pro
import bletl_analysis

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'data')
FP_TESTFILE = pathlib.Path(dir_testfiles, '107-AR_Coryne-AR-2019-04-15-12-39-30.csv')


class TestDOPeakDetection(unittest.TestCase):
    def test_find_peak(self):
        bldata = bletl.parse(FP_TESTFILE)

        x, y = bldata['DO'].get_timeseries('A01')

        c_peak = bletl_analysis.find_do_peak(x, y, delay_a=0.5, threshold_a=70, delay_b=0, threshold_b=80, initial_delay=1)

        self.assertEqual(c_peak, 60)
        return


if __name__ == '__main__':
    unittest.main()
