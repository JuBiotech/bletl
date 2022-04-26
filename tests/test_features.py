import pathlib

import pandas
import pytest

import bletl
from bletl import features

dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, "data")
FP_TESTFILE = pathlib.Path(dir_testfiles, "BLPro", "107-AR_Coryne-AR-2019-04-15-12-39-30.csv")


class TestFeatureExtraction:
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
            last_cycles={
                "A01": 20,
                "B01": 50,
            },
        )
        assert isinstance(extracted_features, pandas.DataFrame)
        assert (
            extracted_features.loc["A01", "pH_x__maximum"]
            == bldata.get_timeseries("pH", "A01", last_cycle=20)[1].max()
        )
        assert (
            extracted_features.loc["B01", "pH_x__sum_values"]
            == bldata.get_timeseries("pH", "B01", last_cycle=50)[1].sum()
        )
        assert (
            extracted_features.loc["C03", "pH_x__sum_values"] == bldata.get_timeseries("pH", "C03")[1].sum()
        )

        # extraction without last_cycles
        extracted_features = features.from_bldata(bldata, extractors, None)
        assert isinstance(extracted_features, pandas.DataFrame)

        # extraction with invalid filterset
        extractors = {"xyz": [features.BSFeatureExtractor()]}
        with pytest.raises(KeyError):
            features.from_bldata(bldata, extractors, None)
        return
