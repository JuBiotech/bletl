import numpy
import pytest
import scipy.stats


try:
    import bletl.growth
    import calibr8
    import pymc3
    from theano.tensor import TensorVariable

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@pytest.fixture
def biomass_calibration():
    """Creates a realistic biomass calibration for testing"""
    # Taken from https://github.com/JuBiotech/calibr8-paper/blob/main/data_and_analysis/processed/biomass.json

    class NonlinearBiomassCalibration(calibr8.BaseLogIndependentAsymmetricLogisticT):
        def __init__(self, *, independent_key:str='X', dependent_key:str='Pahpshmir_1400_BS3_CgWT'):
            super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)

    cmodel = NonlinearBiomassCalibration()
    cmodel.theta_fitted = [
        1.4913711809145784,
        399.9135646512631,
        1.915740229171353,
        500.0691564179732,
        0.743088789680052,
        0.1589557302836782,
        0.007209373982975859,
        30.0
    ]
    yield cmodel


@pytest.fixture
def biomass_curve():
    """Simulates a biomass curve with three different growth rate segments"""
    t = numpy.arange(0, 12, step=10/60)
    mu_true = numpy.ones_like(t) * 0.05
    mu_true[t < 10] = 0.2
    mu_true[t < 8] = 0.4

    # Simulate the biomass concentrations
    X0_true = 0.25
    X = X0_true * numpy.exp(numpy.cumsum(mu_true * numpy.diff(t, prepend=0)))

    yield t, X, mu_true


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Needs optional dependencies.")
class TestGrowthHelpers:
    def test_make_gaussian_random_walk(self):
        with pymc3.Model() as pmodel:
            rv = bletl.growth._make_random_walk(
                "testGRW",
                sigma=0.02,
                length=20,
                student_t=False,
            )
            assert isinstance(rv, TensorVariable)

    def test_make_studentt_random_walk(self):
        with pymc3.Model() as pmodel:
            rv = bletl.growth._make_random_walk(
                "testSTRW",
                sigma=0.02,
                length=20,
                student_t=True,
            )
            assert isinstance(rv, TensorVariable)
        pass

    def test_get_smoothed_mu(self, biomass_curve, biomass_calibration):
        t, X, mu_true = biomass_curve
        loc, scale, df = biomass_calibration.predict_dependent(X)
        # Synthesize data with much smaller scale to make the test less noisy
        # (Yes, the moving window method is easily distracted.)
        bs = scipy.stats.t.rvs(loc=loc, scale=scale / 10, df=df)

        mu = bletl.growth._get_smoothed_mu(t, bs, biomass_calibration)
        assert mu.shape == t.shape
        # Verify based on the mean error w.r.t. the ground truth
        assert numpy.abs(mu - mu_true).mean() < 0.1
        pass


class TestRandomWalkModel:
    def test_fit_mu_t_gaussian(self, biomass_curve, biomass_calibration):
        t, X, mu_true = biomass_curve
        loc, scale, df = biomass_calibration.predict_dependent(X)
        bs = scipy.stats.t.rvs(loc=loc, scale=scale, df=df)

        result = bletl.growth.fit_mu_t(
            t=t, y=bs,
            calibration_model=biomass_calibration,
            student_t=False,
        )
        assert isinstance(result, bletl.growth.GrowthRateResult)
        assert result.mu_map.shape == t.shape
        # Verify based on the mean error w.r.t. the ground truth
        assert numpy.abs(result.mu_map - mu_true).mean() < 0.1
        # No switchpoint detection with Gaussian random walks!
        assert len(result.detected_switchpoints) == 0
        pass

    def test_fit_mu_t_studentt(self, biomass_curve, biomass_calibration):
        t, X, mu_true = biomass_curve
        loc, scale, df = biomass_calibration.predict_dependent(X)

        bs = scipy.stats.t.rvs(loc=loc, scale=scale / 10, df=df)

        result = bletl.growth.fit_mu_t(
            t=t, y=bs,
            calibration_model=biomass_calibration,
            student_t=True,
        )
        assert isinstance(result, bletl.growth.GrowthRateResult)
        assert result.mu_map.shape == t.shape
        # Verify based on the mean error w.r.t. the ground truth
        assert numpy.abs(result.mu_map - mu_true).mean() < 0.1
        pass
