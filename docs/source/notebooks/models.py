import pathlib
import urllib.request

import calibr8


class BLProCDWBackscatterModelV1(calibr8.BaseLogIndependentAsymmetricLogisticT):
    def __init__(self, *, independent_key: str = "X", dependent_key: str = "Pahpshmir_1400_BS3_CgWT"):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


def get_biomass_calibration() -> BLProCDWBackscatterModelV1:
    fp_savefile = pathlib.Path("Quickstart_04_biomass_calibration.json")
    if not fp_savefile.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/JuBiotech/calibr8-paper/main/data_and_analysis/processed/biomass.json",
            fp_savefile,
        )
    return BLProCDWBackscatterModelV1.load(fp_savefile)
