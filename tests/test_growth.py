import numpy
import pytest
import scipy.stats

try:
    import calibr8

    import bletl.growth
    from bletl.growth import pm, pt

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@pytest.fixture
def biomass_calibration():
    """Creates a realistic biomass calibration for testing"""

    class LinearModel(calibr8.BasePolynomialModelT):
        def __init__(self) -> None:
            super().__init__(
                independent_key="X", dependent_key="Pahpshmir_1400_BS3_CgWT", mu_degree=1, scale_degree=0
            )

    cmodel = LinearModel()
    cmodel.theta_fitted = [0, 1, 0.1, 5]
    yield cmodel


@pytest.fixture
def biomass_curve():
    """Simulates a biomass curve with three different growth rate segments"""
    t_data = numpy.arange(0, 12, step=10 / 60)
    t_segments = numpy.mean([t_data[1:], t_data[:-1]], axis=0)

    mu_true = numpy.ones_like(t_segments) * 0.05
    mu_true[t_segments < 10] = 0.2
    mu_true[t_segments < 8] = 0.4

    # Simulate the biomass concentrations
    X0_true = 0.25
    X = numpy.concatenate(
        [
            [X0_true],
            X0_true * numpy.exp(numpy.cumsum(mu_true * numpy.diff(t_data))),
        ]
    )

    assert X.shape == t_data.shape
    assert mu_true.shape == t_segments.shape

    yield t_data, X, mu_true


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Needs optional dependencies.")
class TestGrowthHelpers:
    def test_make_gaussian_random_walk(self):
        with pm.Model() as pmodel:
            rv = bletl.growth._make_random_walk(
                "testGRW",
                init_dist=pm.Normal.dist(),
                sigma=0.02,
                length=20,
                student_t=False,
            )
            assert isinstance(rv, pt.TensorVariable)

    def test_make_studentt_random_walk(self):
        with pm.Model() as pmodel:
            rv = bletl.growth._make_random_walk(
                "testSTRW",
                init_dist=pm.Normal.dist(),
                sigma=0.02,
                length=20,
                student_t=True,
            )
            assert isinstance(rv, pt.TensorVariable)
        pass

    def test_get_smoothed_mu(self, biomass_curve, biomass_calibration):
        t, X, mu_true = biomass_curve
        loc, scale, df = biomass_calibration.predict_dependent(X)
        # Synthesize data with much smaller scale to make the test less noisy
        # (Yes, the moving window method is easily distracted.)
        bs = scipy.stats.t.rvs(loc=loc, scale=scale / 10, df=df)

        mu = bletl.growth._get_smoothed_mu(t, bs, biomass_calibration)
        assert len(mu) == len(t) - 1
        # Verify based on the mean error w.r.t. the ground truth
        assert numpy.abs(mu - mu_true).mean() < 0.1
        pass


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Needs optional dependencies.")
class TestRandomWalkModel:
    def test_fit_mu_t_gaussian(self, biomass_curve, biomass_calibration):
        t, X, mu_true = biomass_curve
        loc, scale, df = biomass_calibration.predict_dependent(X)
        bs = scipy.stats.t.rvs(loc=loc, scale=scale, df=df)

        result = bletl.growth.fit_mu_t(
            t=t,
            y=bs,
            calibration_model=biomass_calibration,
            student_t=False,
            drift_scale=0.01,
        )
        assert isinstance(result, bletl.growth.GrowthRateResult)
        assert len(result.mu_map) == len(t) - 1
        assert len(result.t_data) == len(t)
        assert len(result.t_segments) == len(t) - 1

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
            t=t,
            y=bs,
            calibration_model=biomass_calibration,
            student_t=True,
            mu_prior=0.4,
            drift_scale=0.01,
        )
        assert isinstance(result, bletl.growth.GrowthRateResult)
        assert len(result.mu_map) == len(t) - 1

        # Verify based on the mean error w.r.t. the ground truth
        assert numpy.abs(result.mu_map - mu_true).mean() < 0.1

        # There were two switchpoints in the data
        assert set(result.switchpoints) == {8.0, 10.0}
        assert set(result.known_switchpoints) == set()
        assert set(result.detected_switchpoints) == {8.0, 10.0}
        pass

    def test_fit_mu_t_studentt_with_known_switchpoints(self, biomass_curve, biomass_calibration):
        t, X, mu_true = biomass_curve
        loc, scale, df = biomass_calibration.predict_dependent(X)

        bs = scipy.stats.t.rvs(loc=loc, scale=scale / 10, df=df)

        result = bletl.growth.fit_mu_t(
            t=t,
            y=bs,
            calibration_model=biomass_calibration,
            student_t=True,
            # Let one of the real switchpoints be known already.
            # The 7.99 switchpoint replaces the autodetection at 8.0.
            switchpoints=[4.0, 7.99],
            mu_prior=0.4,
            drift_scale=0.01,
        )
        assert len(result.mu_map) == len(t) - 1

        # Verify based on the mean error w.r.t. the ground truth
        assert numpy.abs(result.mu_map - mu_true).mean() < 0.1

        # There were two switchpoints in the data
        assert set(result.switchpoints) == {4.0, 7.99, 10.0}
        assert set(result.known_switchpoints) == {4.0, 7.99}
        assert set(result.detected_switchpoints) == {10.0}
        pass

    @pytest.mark.parametrize("student_t", [False, True])
    def test_custom_mu_zero(self, biomass_calibration, student_t):
        t = numpy.arange(0, 10, 0.1)
        X = 0.25 * numpy.exp(t * 0.42)
        loc, scale, df = biomass_calibration.predict_dependent(X)

        rng = numpy.random.RandomState(2022)
        bs = scipy.stats.t.rvs(loc=loc, scale=scale, df=df, random_state=rng)

        numpy.random.seed(2022)
        result = bletl.growth.fit_mu_t(
            t=t,
            y=bs,
            calibration_model=biomass_calibration,
            student_t=student_t,
            mu_prior=0.42,
            drift_scale=0.01,
        )
        assert numpy.mean(result.mu_map - 0.42) < 0.1
        pass
