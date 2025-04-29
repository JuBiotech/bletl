"""Contains unit tests for the `bletl` package"""
import datetime
import pathlib

import numpy
import pandas
import pytest

import bletl
from bletl import (
    InvalidLotNumberError,
    LotInformationMismatch,
    NoMeasurementData,
    core,
    parsing,
    utils,
)
from bletl.parsing import bl1, blpro

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, "data")

BL1_files = [
    pathlib.Path(dir_testfiles, "BL1", "NT_1400rpm_30C_BS15_5min_20180618_102917.csv"),
    pathlib.Path(dir_testfiles, "BL1", "JH_ShakerSteps_20170302_070206.csv"),
    pathlib.Path(dir_testfiles, "BL1", "rj-cg-res_20170927_084112.csv"),
    pathlib.Path(dir_testfiles, "BL1", "incremental", "NT_1400rpm_30C_BS15_5min_20180503_132133.csv"),
]

BL1_fragment_files = [
    pathlib.Path(dir_testfiles, "BL1", "fragments", "fragment0.csv"),
    pathlib.Path(dir_testfiles, "BL1", "fragments", "fragment1.csv"),
    pathlib.Path(dir_testfiles, "BL1", "fragments", "fragment2.csv"),
]

BL1_files_without_calibration_info = [
    pathlib.Path(dir_testfiles, "BL1", "NT_1400_BS20BS10_15min_20160222_151645.csv"),
]

not_a_bl_file = pathlib.Path(dir_testfiles, "BL1", "incremental", "C42.tmp")

BL2_files = list(pathlib.Path(dir_testfiles, "BLII").iterdir())
BLPro_files = list(pathlib.Path(dir_testfiles, "BLPro").iterdir())
calibration_test_file = pathlib.Path(dir_testfiles, "BLPro", "18-FZJ-Test2--2018-02-07-10-01-11.csv")
incompatible_file = pathlib.Path(dir_testfiles, "incompatible_files", "BL2-file-saved-with-biolection.csv")

calibration_test_file_pro = pathlib.Path(dir_testfiles, "BLPro", "18-FZJ-Test2--2018-02-07-10-01-11.csv")
calibration_test_file_pro2 = pathlib.Path(
    dir_testfiles, "BLPro", "8-HM_CoryneBatch-HM-2018-04-11-14-52-54.csv"
)

calibration_test_file = pathlib.Path(
    dir_testfiles, "BL1", "NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv"
)
calibration_test_cal_data = {
    "cal_0": 65.91,
    "cal_100": 40.60,
    "phi_min": 57.45,
    "phi_max": 18.99,
    "pH_0": 6.46,
    "dpH": 0.56,
}
calibration_test_times = {"BS10": (52, "D03", 10.5221)}
calibration_test_values = {
    "BS10": (5, "A04", 11.7175),
    "DO": (13, "A05", 99.4285),
    "pH": (39, "D08", 7.06787),
    "GFP75": (81, "F07", 216.99),
}
calibration_test_lot_number = 1515
calibration_test_temp = 30

file_with_lot_info = pathlib.Path(
    dir_testfiles, "BL1", "example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv"
)
file_with_no_measurements = pathlib.Path(
    dir_testfiles, "BL1", "broken_or_incomplete", "file_with_no_measurements.csv"
)


class TestParserSelection:
    @pytest.mark.parametrize("fp", BL1_files)
    def test_selects_parsers(self, fp):
        parser = bletl.get_parser(fp)
        assert isinstance(parser, core.BLDParser)
        assert isinstance(parser, parsing.bl1.BioLector1Parser)
        return

    def test_fail_on_unsupported(self):
        with pytest.raises(ValueError):
            bletl.get_parser(not_a_bl_file)
        return

    @pytest.mark.parametrize("fp", BLPro_files)
    def test_selects_parsers_pro(self, fp):
        parser = bletl.get_parser(fp)
        assert isinstance(parser, core.BLDParser)
        assert isinstance(parser, parsing.blpro.BioLectorProParser)
        return

    @pytest.mark.parametrize("fp", BL2_files)
    def test_selects_parsers_ii(self, fp):
        parser = bletl.get_parser(fp)
        assert isinstance(parser, core.BLDParser)
        assert isinstance(parser, parsing.blpro.BioLectorProParser)
        return

    def test_incompatible_file_detecion(self):
        with pytest.raises(bletl.IncompatibleFileError):
            bletl.get_parser(incompatible_file)
        with pytest.raises(bletl.IncompatibleFileError):
            bletl.get_parser(dir_testfiles / "incompatible_files" / "other.zip")

    def test_xt_file_recognition(self):
        fp = dir_testfiles / "BLXT" / "4_BG_test.zip"
        with pytest.raises(NotImplementedError, match=r"XT.*?version: 1\.0\.0"):
            bletl.get_parser(fp)
        pass


class TestUtils:
    def test_last_well_in_cycle(self):
        measurements = pandas.DataFrame(
            data={
                "cycle": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "filterset": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] * 2,
                "well": ["A01", "A02", "A03", "A04"] * 3 * 2,
            }
        )
        last_well = utils._last_well_in_cycle(measurements)
        assert last_well == "A04"

        # cut off the last two measurements
        measurements = measurements.iloc[:-2]
        return

    def test_last_full_cycle(self):
        measurements = pandas.DataFrame(
            data={
                "cycle": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "filterset": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] * 2,
                "well": ["A01", "A02", "A03", "A04"] * 3 * 2,
            }
        )

        last_cycle = utils._last_full_cycle(measurements)
        assert last_cycle == 2

        # cut off the last two measurements
        measurements = measurements.iloc[:-2]

        last_cycle = utils._last_full_cycle(measurements)
        assert last_cycle == 1
        return

    def test_cal_info_parsing(self):
        example = "1724-hc-Temp37"
        lot_number, temp = utils._parse_calibration_info(example)
        assert lot_number == 1724
        assert temp == 37
        return


class TestBL1Parsing:
    @pytest.mark.parametrize("fp", BL1_files)
    def test_splitting(self, fp):
        with open(fp, "r", encoding="latin-1") as f:
            lines = f.readlines()

        headerlines, data = parsing.bl1.split_header_data(fp)

        assert len(headerlines) + len(data) == len(lines)
        return

    @pytest.mark.parametrize("fp", BL1_files)
    def test_parsing(self, fp):
        data = bletl.parse(fp)

        assert isinstance(data.model, core.BioLectorModel)
        assert data.model == core.BioLectorModel.BL1
        assert isinstance(data.metadata, dict)
        assert isinstance(data.environment, pandas.DataFrame)
        assert isinstance(data.comments, pandas.DataFrame)
        assert isinstance(data.measurements, pandas.DataFrame)
        assert isinstance(data.references, pandas.DataFrame)
        assert isinstance(data.wells, tuple)
        assert len(data.wells) == 48
        return

    def test_concat_parsing(self):
        filepaths = BL1_fragment_files
        data = bletl.parse(filepaths)
        assert isinstance(data.metadata, dict)
        assert isinstance(data.environment, pandas.DataFrame)
        assert isinstance(data.comments, pandas.DataFrame)
        assert isinstance(data.measurements, pandas.DataFrame)
        assert isinstance(data.references, pandas.DataFrame)
        return

    def test_incomplete_cycle_drop(self):
        filepath = BL1_files[2]
        data = bletl.parse(filepath, drop_incomplete_cycles=False)
        assert data.measurements.index[-1] == (3, 179, "C08")

        data = bletl.parse(filepath, drop_incomplete_cycles=True)
        assert data.measurements.index[-1] == (3, 178, "F01")
        return

    def test_temp_setpoint_parsing(self):
        fp = pathlib.Path(dir_testfiles, "BL1", "NT_1400rpm_30C_BS15_5min_20180618_102917.csv")
        data = bletl.parse(fp)
        df = data.environment
        temps = set(df["temp_setpoint"].unique())
        assert temps == set([30.0, 25.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0])
        # test some individual values
        assert list(df.loc[numpy.isclose(df["time"], 0.00778), "temp_setpoint"])[0] == 30.0
        assert list(df.loc[numpy.isclose(df["time"], 55.30764), "temp_setpoint"])[0] == 37.0
        assert list(df.loc[numpy.isclose(df["time"], 55.44867), "temp_setpoint"])[0] == 38.0
        assert list(df.loc[numpy.isclose(df["time"], 126.74621), "temp_setpoint"])[0] == 40.0
        return

    def test_shaker_setpoint_parsing(self):
        fp = pathlib.Path(dir_testfiles, "BL1", "JH_ShakerSteps_20170302_070206.csv")
        data = bletl.parse(fp)
        df = data.environment
        rpms = set(df["shaker_setpoint"].unique())
        assert rpms == set(
            [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
        )
        # test some individual values
        assert list(df.loc[numpy.isclose(df["time"], 0.06265), "shaker_setpoint"])[0] == 500.0
        assert list(df.loc[numpy.isclose(df["time"], 2.49103), "shaker_setpoint"])[0] == 900.0
        assert list(df.loc[numpy.isclose(df["time"], 2.50949), "shaker_setpoint"])[0] == 1000.0
        assert list(df.loc[numpy.isclose(df["time"], 5.65632), "shaker_setpoint"])[0] == 500.0
        return

    def test_get_timeseries(self):
        fp = pathlib.Path(dir_testfiles, "BL1", "NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv")
        data = bletl.parse(fp, lot_number=calibration_test_lot_number, temp=30)
        x, y = data["pH"].get_timeseries("A03")
        assert len(x) == len(y)
        assert len(x) == 103
        assert numpy.sum(x) == 1098.50712
        assert numpy.sum(y) == 746.2625060506602
        return

    def test_get_timeseries_last_cycle(self):
        fp = pathlib.Path(dir_testfiles, "BL1", "NT_1200rpm_30C_DO-GFP75-pH-BS10_12min_20171221_121339.csv")
        data = bletl.parse(fp, lot_number=calibration_test_lot_number, temp=30)

        # default to all
        x, y = data["pH"].get_timeseries("A03")
        assert len(x) == len(y)
        assert len(x) == 103

        # invalid values
        with pytest.raises(ValueError):
            data["pH"].get_timeseries("A03", last_cycle=-1)

        with pytest.raises(ValueError):
            data["pH"].get_timeseries("A03", last_cycle=0)

        # valid settings
        x, y = data["pH"].get_timeseries("A03", last_cycle=1)
        assert len(x) == 1
        assert len(y) == 1

        x, y = data["pH"].get_timeseries("A03", last_cycle=50)
        assert len(x) == 50
        assert len(y) == 50

        # more than available
        x, y = data["pH"].get_timeseries("A03", last_cycle=200)
        assert len(x) == 103
        assert len(y) == 103
        pass

    def test_get_unified_dataframe(self):
        fp = pathlib.Path(
            dir_testfiles, "BL1", "example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv"
        )
        data = bletl.parse(fp)
        unified_df = data["BS20"].get_unified_dataframe()
        assert isinstance(unified_df, pandas.DataFrame)
        numpy.testing.assert_approx_equal(unified_df.index[2], 0.35313, significant=4)
        idx = unified_df.index.get_indexer([5], method="nearest")
        col = unified_df.columns.get_indexer(["A05"])
        values = unified_df.iloc[idx, col].to_numpy()
        numpy.testing.assert_approx_equal(values, 63.8517, significant=6)

    def test_get_narrow_data(self):
        fp = pathlib.Path(
            dir_testfiles, "BL1", "example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv"
        )
        data = bletl.parse(fp)
        narrow_data = data.get_narrow_data()
        assert isinstance(narrow_data, pandas.DataFrame)
        numpy.testing.assert_approx_equal(narrow_data.loc[52884, "value"], 102.835, significant=6)

    def test_get_unified_narrow_data(self):
        fp = pathlib.Path(
            dir_testfiles, "BL1", "example_with_cal_data_NT_1400rpm_30C_BS20-pH-DO_10min_20180607_115856.csv"
        )
        data = bletl.parse(fp)

        unified_narrow_data_1 = data.get_unified_narrow_data()
        assert isinstance(unified_narrow_data_1, pandas.DataFrame)
        numpy.testing.assert_approx_equal(unified_narrow_data_1.loc[21771, "pH"], 7.405, significant=4)

        unified_narrow_data_2 = data.get_unified_narrow_data(source_filterset="DO", source_well="B04")
        numpy.testing.assert_approx_equal(unified_narrow_data_2.loc[0, "time"], 0.09409, significant=5)

        with pytest.raises(KeyError):
            data.get_unified_narrow_data(source_filterset="machine_that_goes_ping")

        with pytest.raises(KeyError):
            data.get_unified_narrow_data(source_well="O9000")

    def test_NoMeasurements_Warning(self):
        with pytest.warns(NoMeasurementData):
            bletl.parse(file_with_no_measurements)

    def test_get_unified_dataframe_well_selection(self):
        # Create test data
        time_df = pandas.DataFrame({"A01": [0.0, 1.0, 2.0], "A02": [0.0, 1.0, 2.0]})
        value_df = pandas.DataFrame({"A01": [1.0, 2.0, 3.0], "A02": [1.5, 2.5, 3.5]})

        fts = bletl.FilterTimeSeries(time_df, value_df)

        # Test with valid well ID - should return DataFrame with correct time values
        df = fts.get_unified_dataframe(well="A01")
        numpy.testing.assert_array_equal(df.index.values, [0.0, 1.0, 2.0])

        # Test with non-existent well ID - should raise KeyError
        with pytest.raises(KeyError, match="Could not find well id"):
            fts.get_unified_dataframe(well="X99")

        # Test with None
        df_default = fts.get_unified_dataframe(well=None)
        numpy.testing.assert_array_equal(df_default.index.values, [0.0, 1.0, 2.0])


class TestBL1Calibration:
    def test_calibration_data_type(self):
        data = bletl.parse(
            calibration_test_file, lot_number=calibration_test_lot_number, temp=calibration_test_temp
        )

        for _, item in data.items():
            assert isinstance(item, core.FilterTimeSeries)

        return

    def test_single_file_with_lot(self):
        data = bletl.parse(
            calibration_test_file, lot_number=calibration_test_lot_number, temp=calibration_test_temp
        )

        for key, (cycle, well, value) in calibration_test_times.items():
            numpy.testing.assert_approx_equal(data[key].time.loc[cycle, well], value, significant=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            numpy.testing.assert_approx_equal(data[key].value.loc[cycle, well], value, significant=4)

        return

    def test_single_file_with_caldata(self):
        data = bletl.parse(calibration_test_file, **calibration_test_cal_data)

        for key, (cycle, well, value) in calibration_test_times.items():
            numpy.testing.assert_approx_equal(data[key].time.loc[cycle, well], value, significant=4)

        for key, (cycle, well, value) in calibration_test_values.items():
            numpy.testing.assert_approx_equal(data[key].value.loc[cycle, well], value, significant=4)

        return

    def test_fragments_with_lot(self):
        filepaths = BL1_fragment_files
        data = bletl.parse(filepaths, lot_number=1846, temp=37)

        numpy.testing.assert_approx_equal(data["DO"].value.loc[666, "F07"], 12.1887, significant=6)
        numpy.testing.assert_approx_equal(data["pH"].value.loc[507, "E06"], 6.6435, significant=5)

        return

    def test_fragments_with_cal_data(self):
        filepaths = BL1_fragment_files
        data = bletl.parse(
            filepaths=filepaths,
            cal_0=71.93,
            cal_100=38.64,
            phi_min=55.36,
            phi_max=11.91,
            pH_0=6.05,
            dpH=0.53,
            drop_incomplete_cycles=True,
        )

        numpy.testing.assert_approx_equal(data["DO"].value.loc[666, "F07"], 12.1887, significant=6)
        numpy.testing.assert_approx_equal(data["pH"].value.loc[507, "E06"], 6.6435, significant=5)

        return

    def test_mismatch_warning(self):
        with pytest.warns(LotInformationMismatch):
            bletl.parse(file_with_lot_info, lot_number=1818, temp=37)
        return


class TestOnlineMethods:
    def test_get_calibration_dict(self):
        cal_dict_fetched = bl1.fetch_calibration_data(1515, 30)
        assert cal_dict_fetched == calibration_test_cal_data
        return

    def test_invalid_lot_number(self):
        with pytest.raises(InvalidLotNumberError):
            bl1.fetch_calibration_data(99, 99)
        return

    def test_download_calibration_data(self):
        utils.download_calibration_data()
        return


class TestBL2Parsing:
    @pytest.mark.parametrize("fp", BL2_files)
    def test_parse_metadata_data(self, fp):
        metadata, data = parsing.blpro.parse_metadata_data(fp)

        assert isinstance(metadata, dict)
        assert isinstance(data, pandas.DataFrame)
        pass

    @pytest.mark.parametrize("fp", BL2_files)
    def test_parsing(self, fp):
        data = bletl.parse(fp)
        assert isinstance(data, dict)
        assert isinstance(data.wells, tuple)
        assert len(data.wells) == 48
        pass


class TestBLProParsing:
    @pytest.mark.parametrize("fp", BLPro_files)
    def test_parse_metadata_data(self, fp):
        metadata, data = parsing.blpro.parse_metadata_data(fp)

        assert isinstance(metadata, dict)
        assert isinstance(data, pandas.DataFrame)

        # 👇 Regression check against https://github.com/JuBiotech/bletl/issues/8
        filtersets = bletl.parsing.blpro.extract_filtersets(metadata)
        assert "01_reference_gain_Biomass" not in metadata["process"]
        return

    @pytest.mark.parametrize("fp", BLPro_files)
    def test_parsing(self, fp):
        data = bletl.parse(fp)
        assert isinstance(data, dict)
        return

    def test_parse_metadata_data_new_format(self):
        fp = pathlib.Path(dir_testfiles, "BLPro", "new_metadata_format.csv")
        metadata, data = parsing.blpro.parse_metadata_data(fp)
        assert isinstance(metadata, dict)
        assert isinstance(data, pandas.DataFrame)
        pass

    def test_parse_new_format(self):
        fp = pathlib.Path(dir_testfiles, "BLPro", "new_metadata_format.csv")
        bldata = bletl.parse(fp)
        assert "BS3" in bldata
        t, y = bldata["BS3"].get_timeseries("A01")
        numpy.testing.assert_array_almost_equal(t, [0.01111111, 0.08888889])
        numpy.testing.assert_array_almost_equal(y, [4.19, 1.96])
        pass

    def test_parse_with_concat(self):
        data = bletl.parse(
            filepaths=[
                pathlib.Path(dir_testfiles, "BLPro", "224-MO_Coryne--2019-07-12-16-54-30.csv"),
                pathlib.Path(dir_testfiles, "BLPro", "226-MO_Coryne--2019-07-12-17-38-02.csv"),
            ]
        )
        assert isinstance(data, bletl.BLData)
        assert data.metadata["date_start"] == datetime.datetime(2019, 7, 12, 16, 54, 30)
        assert data.metadata["date_end"] is None
        numpy.testing.assert_array_equal(data["BS5"].time.index, numpy.arange(1, 4 + 254 + 1))
        numpy.testing.assert_array_almost_equal(
            data["BS5"].time["A01"].iloc[:5], [0.013056, 0.179444, 0.346111, 0.512778, 0.738611]
        )
        return

    def test_parse_file_with_defects(self):
        # this file has some broken & duplicate data line lines 25857-25877
        bletl.parse(pathlib.Path(dir_testfiles, "BLPro", "line_duplication.csv"))
        pass

    @pytest.mark.filterwarnings("ignore:cycle 2 filterset 02")
    def test_drop_non_monotonically_increasing_time_filtersets(self):
        """This happens when line/block defects accidentally result in a parseable CSV."""
        with pytest.warns(UserWarning, match="cycle 2 filterset 02"):
            bldata = bletl.parse(dir_testfiles / "BLPro" / "non_monotonic_time.csv")
            assert len(bldata["BS5"].time) == 4
            assert len(bldata["pH"].time) == 3
            assert 2 not in bldata["pH"].time.index
            assert len(bldata["DO"].time) == 4
        pass

    def test_refoverld_issue12(self):
        with pytest.warns(UserWarning, match=r"cycles \[269, 636\].*REFOVERLD"):
            bldata = bletl.parse(dir_testfiles / "BLPro" / "issue12.csv")
        for fs, n in zip(["BS3", "DO", "pH"], [682, 684, 684]):
            t, y = bldata.get_timeseries(fs, "A01")
            assert len(t) == n
            assert len(y) == n
        pass

    def test_issue24(self):
        bldata = bletl.parse(dir_testfiles / "BLPro" / "issue24.csv")
        assert "BS1" in bldata
        assert "BS3" in bldata
        assert "pH" in bldata
        assert "DO" in bldata
        assert "Fluorescence5" not in bldata
        assert "Fluorescence9" not in bldata
        pass


class TestBLProCalibration:
    def test_filtertimeseries_presence(self):
        bd = bletl.parse(calibration_test_file_pro)
        assert isinstance(bd, dict)
        assert "BS2" in bd
        assert "BS5" in bd
        assert "pH" in bd
        assert "DO" in bd
        return

    def test_correct_well_association(self):
        bd = bletl.parse(calibration_test_file_pro)

        # F01
        x, y = bd["pH"].get_timeseries("F01")
        assert numpy.allclose(x[:3], [0.07972222, 0.16166667, 0.24472222])
        assert numpy.allclose(y[:3], [8.15, 7.93, 7.78])

        # D02
        x, y = bd["DO"].get_timeseries("D02")
        assert numpy.allclose(x[:3], [0.09166667, 0.17305556, 0.25611111])
        assert numpy.allclose(y[:3], [92.09, 93.84, 94.65])
        return

    def test_calibration_with_lot_number(self):
        org = bletl.parse(calibration_test_file_pro2)
        with_lot = bletl.parse(calibration_test_file_pro2, lot_number=1724, temp=30)

        random_cycle = numpy.random.randint(len(org["pH"].time.index)) + 1

        numpy.testing.assert_allclose(
            org["pH"].value.loc[random_cycle],
            with_lot["pH"].value.loc[random_cycle],
            rtol=0.02,
        )

        numpy.testing.assert_allclose(
            org["DO"].value.loc[random_cycle],
            with_lot["DO"].value.loc[random_cycle],
            rtol=0.02,
        )

    def test_calibration_with_parameters(self):
        org = bletl.parse(calibration_test_file_pro2)
        with_p = bletl.parse(
            calibration_test_file_pro2,
            cal_0=71.8655,
            cal_100=42.9188,
            phi_min=64.248,
            phi_max=19.039,
            pH_0=6.667,
            dpH=0.4878,
        )

        random_cycle = numpy.random.randint(len(org["pH"].time.index)) + 1

        numpy.testing.assert_allclose(
            org["pH"].value.loc[random_cycle],
            with_p["pH"].value.loc[random_cycle],
            rtol=0.02,
        )

        numpy.testing.assert_allclose(
            org["DO"].value.loc[random_cycle],
            with_p["DO"].value.loc[random_cycle],
            rtol=0.02,
        )


class TestDataEquivalence:
    def test_model(self):
        data = bletl.parse(BLPro_files[0])

        assert isinstance(data.model, core.BioLectorModel)
        assert data.model == core.BioLectorModel.BLPro
        return

    def test_environment(self):
        d_1 = bletl.parse(BL1_files[0], lot_number=1818, temp=30)
        d_p = bletl.parse(BLPro_files[0])

        assert list(d_1.environment.columns) == list(d_p.environment.columns)
        return

    def test_filtersets(self):
        d_1 = bletl.parse(BL1_files[0], lot_number=1818, temp=30)
        d_p = bletl.parse(BLPro_files[0])

        assert list(d_1.filtersets.columns) == list(d_p.filtersets.columns)
        return

    def test_references(self):
        d_1 = bletl.parse(BL1_files[0], lot_number=1818, temp=30)
        d_p = bletl.parse(BLPro_files[0])

        assert list(d_1.references.columns) == list(d_p.references.columns)
        return

    def test_measurements(self):
        d_1 = bletl.parse(BL1_files[0], lot_number=1818, temp=30)
        d_p = bletl.parse(BLPro_files[0])

        assert list(d_1.measurements.columns) == list(d_p.measurements.columns)
        return

    def test_comments(self):
        d_1 = bletl.parse(BL1_files[0], lot_number=1818, temp=30)
        d_p = bletl.parse(BLPro_files[0])

        assert list(d_1.comments.columns) == list(d_p.comments.columns)
        return


class TestBLProMF:
    def test_well_num_mappings(self):
        # C-style counts right then down
        assert blpro._MF_WELL_NUMC_TO_ID[0] == "C01"
        assert blpro._MF_WELL_NUMC_TO_ID[1] == "C02"
        assert blpro._MF_WELL_NUMC_TO_ID[7] == "C08"
        assert blpro._MF_WELL_NUMC_TO_ID[8] == "D01"
        assert blpro._MF_WELL_NUMC_TO_ID[31] == "F08"
        # F-style counts down then right
        assert blpro._MF_WELL_NUMF_TO_ID[0] == "C01"
        assert blpro._MF_WELL_NUMF_TO_ID[1] == "D01"
        assert blpro._MF_WELL_NUMF_TO_ID[2] == "E01"
        assert blpro._MF_WELL_NUMF_TO_ID[3] == "F01"
        assert blpro._MF_WELL_NUMF_TO_ID[4] == "C02"
        assert blpro._MF_WELL_NUMF_TO_ID[31] == "F08"
        # Measurement order style is 1-based and goes left/right vice versa
        assert blpro._MF_WELL_NUMM_TO_ID[1] == "C01"
        assert blpro._MF_WELL_NUMM_TO_ID[2] == "C02"
        assert blpro._MF_WELL_NUMM_TO_ID[8] == "C08"
        assert blpro._MF_WELL_NUMM_TO_ID[9] == "D08"
        assert blpro._MF_WELL_NUMM_TO_ID[16] == "D01"
        assert blpro._MF_WELL_NUMM_TO_ID[32] == "F01"
        pass

    def test_issue_38(self):
        fp = dir_testfiles / "BLPro" / "18-FZJ-Test2--2018-02-07-10-01-11.csv"
        bldata = bletl.parse(fp)
        assert bldata.fluidics.index.name == "well"
        assert bldata.module.index.names == ["well", "valve", "cycle"]
        assert bldata.valves.index.names == ["well", "valve", "cycle"]
        # Check some initial and final well volumes against values shown in the BioLection
        assert bldata.fluidics.loc["C01", "volume"][0] == 800
        assert bldata.fluidics.loc["C01", "volume"][-1] == 1201.776
        assert bldata.fluidics.loc["D01", "volume"][-1] == 1204.892
        assert bldata.fluidics.loc["D02", "volume"][-1] == 954.68
        assert bldata.fluidics.loc["E06", "volume"][-1] == 913.16
        assert bldata.fluidics.loc["F01", "volume"][-1] == 1202.719
        pass

    def test_fluidics_source(self):
        fp = dir_testfiles / "BLPro" / "18-FZJ-Test2--2018-02-07-10-01-11.csv"
        bldata = bletl.parse(fp)
        assert isinstance(bldata.fluidics["reservoir"][0], bletl.FluidicsSource)
        pass
