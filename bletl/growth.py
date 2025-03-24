import importlib.util
import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import arviz
import calibr8
import numpy
import pymc as pm
import pytensor.tensor as pt
from packaging import version

_log = logging.getLogger(__file__)


class GrowthRateResult:
    """Represents the result of applying the µ(t) model to one dataset."""

    def __init__(
        self,
        *,
        t_data: Union[Sequence[float], numpy.ndarray],
        t_segments: Union[Sequence[float], numpy.ndarray],
        y: Union[Sequence[float], numpy.ndarray],
        calibration_model: calibr8.CalibrationModel,
        switchpoints: Dict[float, str],
        pmodel: pm.Model,
        theta_map: Dict[str, numpy.ndarray],
    ):
        """Creates a result object of a growth rate analysis.

        Parameters
        ----------
        t_data : numpy.ndarray
            time vector of data timepoints
        t_segments : numpy.ndarray
            time vector of growth rate segment midpoints
        y : numpy.ndarray
            backscatter vector
        switchpoints : dict
            maps switchpoint times to labels
        pmodel : pymc.Model
            the PyMC model underlying the analysis
        theta_map : dict
            the PyMC MAP estimate
        """
        self._t_data = numpy.asarray(t_data)
        self._t_segments = numpy.asarray(t_segments)
        self._y = numpy.asarray(y)
        self._switchpoints = switchpoints
        self.calibration_model = calibration_model
        self._pmodel = pmodel
        self._theta_map = theta_map
        self._idata = None
        super().__init__()

    @property
    def t_data(self) -> numpy.ndarray:
        """Vector of data timepoints."""
        return self._t_data

    @property
    def t_segments(self) -> numpy.ndarray:
        """Vector of growth rate segment time mid-points."""
        return self._t_segments

    @property
    def y(self) -> numpy.ndarray:
        """Vector of backscatter observations."""
        return self._y

    @property
    def switchpoints(self) -> Dict[float, str]:
        """Dictionary (by time) of known and detected switchpoints."""
        return self._switchpoints

    @property
    def known_switchpoints(self) -> Tuple[float, ...]:
        """Time values of previously known switchpoints in the model."""
        return tuple(t for t, label in self.switchpoints.items() if label != "detected")

    @property
    def detected_switchpoints(self) -> Tuple[float, ...]:
        """Time values of switchpoints that were autodetected from the fit."""
        return tuple(t for t, label in self.switchpoints.items() if label == "detected")

    @property
    def pmodel(self) -> pm.Model:
        """The PyMC model underlying this analysis."""
        return self._pmodel

    @property
    def theta_map(self) -> Dict[str, numpy.ndarray]:
        """MAP estimate of the model parameters."""
        return self._theta_map

    @property
    def idata(self) -> Optional[arviz.InferenceData]:
        """ArviZ InferenceData object of the MCMC trace."""
        return self._idata

    @property
    def mu_map(self) -> numpy.ndarray:
        """MAP estimate of the growth rates in segments between data points."""
        return self.theta_map["mu_t"]

    @property
    def x_map(self) -> numpy.ndarray:
        """MAP estimate of the biomass curve."""
        return self.theta_map["X"]

    @property
    def mu_mcmc(self) -> Optional[numpy.ndarray]:
        """Posterior samples of growth rates in segments between data points."""
        if not self.idata:
            return None
        assert hasattr(self.idata, "posterior")
        return self.idata.posterior.mu_t.stack(sample=("chain", "draw")).values.T

    @property
    def x_mcmc(self) -> Optional[numpy.ndarray]:
        """Posterior samples of biomass curve."""
        if self.idata is None:
            return None
        assert hasattr(self.idata, "posterior")
        return self.idata.posterior["X"].stack(sample=("chain", "draw")).T

    def sample(self, **kwargs) -> None:
        """Runs MCMC sampling with default settings on the growth model.

        Parameters
        ----------
        **sample_kwargs
            optional keyword-arguments to pymc.sample(...) to override defaults
        """
        if importlib.util.find_spec("nutpie"):
            sample_kwargs = dict(
                return_inferencedata=True,
                target_accept=0.95,
                nuts_sampler="nutpie",
                init="adapt_diag",
                init_means=self.theta_map,
                tune=500,
                draws=500,
            )
        else:
            sample_kwargs = dict(
                return_inferencedata=True,
                target_accept=0.95,
                init="adapt_diag",
                initvals=self.theta_map,
                tune=500,
                draws=500,
            )
        sample_kwargs.update(kwargs)
        with self.pmodel:
            self._idata = pm.sample(**sample_kwargs)
        return


def _make_random_walk(
    name: str,
    *,
    init_dist: pt.TensorVariable,
    mu: float = 0,
    sigma: float,
    nu: float = 1,
    length: int,
    student_t: bool,
    initval: Optional[numpy.ndarray] = None,
    dims: Optional[str] = None,
):
    """Create a random walk with either a Normal or Student-t distribution.

    For some PyMC versions and for Student-t distributed random walks,
    the distribution is created from a cumsum of a N-dimensional random variable.

    Parameters
    ----------
    name : str
        Name of the random walk variable.
    init_dist
        A random variable to use as the prior for innovations.
    mu : float, array-like
        Mean of the random walk.
    sigma : float, array-like
        Standard deviation (Normal) or scale (StudentT) parameter.
    nu : float, array-like
        Degree of freedom for the StudentT distribution - only used when `student_t == True`.
    length : int
        Number of steps in the random walk.
    student_t : bool
        If `True` a `pymc.Deterministic` of a StudentT-random walk is created.
        Otherwise a `GaussianRandomWalk` is created.
    initval : numpy.ndarray
        Initial values for the RandomWalk variable.
        If set, PyMC uses these values as start points for MAP optimization and MCMC sampling.
    dims
        Optional dims to be forwarded to the `RandomWalk`.

    Returns
    -------
    random_walk : TensorVariable
        The tensor variable of the random walk.
    """
    pmversion = version.parse(pm.__version__)

    if pmversion < version.parse("5.0.0"):
        raise NotImplementedError("PyMC versions <5.0.0 are no longer supported.")

    if student_t:
        innov_dist = pm.StudentT.dist(mu=mu, sigma=sigma, nu=nu)
    else:
        innov_dist = pm.Normal.dist(mu=mu, sigma=sigma)

    rw = pm.RandomWalk(
        name,
        init_dist=init_dist,
        innovation_dist=innov_dist,
        steps=length - 1,
        initval=initval,
        dims=dims,
    )
    return rw


def _get_smoothed_mu(
    t: Sequence[float],
    y: Sequence[float],
    cm_cdw: calibr8.CalibrationModel,
    *,
    clip: float = 0.5,
) -> numpy.ndarray:
    """Calculate a rough estimate of the specific growth rate from smoothed observations.

    Parameters
    ----------
    t : numpy.ndarray
        Timepoints
    y : numpy.ndarray
        Observations in measurement units
    cm_cdw : calibr8.CalibrationModel
        Calibration model that predicts measurement units from biomass concentration in g/L.
    clip : float
        Maximum/minimum growth rate for clipping.

    Returns
    -------
    mu : numpy.ndarray
        A vector of specific growth rates.
    """
    # apply moving average to reduce backscatter noise
    yarr = numpy.convolve(y, numpy.ones(5) / 5, "same")

    # convert to biomass
    X = cm_cdw.predict_independent(yarr)

    # calculate growth rate
    dX = numpy.diff(X)
    dt = numpy.diff(t)
    Xsegment = numpy.mean([X[1:], X[:-1]], axis=0)
    mu = (dX / dt) / Xsegment

    # clip growth rate into a realistic interval
    mu = numpy.clip(mu, -clip, clip)

    # smooth again to reduce peaking
    mu = numpy.convolve(mu, numpy.ones(5) / 5, "same")

    # Replace NaNs that can show up with non-linear calibration models
    mu[numpy.isnan(mu)] = 0
    return mu


def fit_mu_t(
    t: Sequence[float],
    y: Sequence[float],
    calibration_model: calibr8.CalibrationModel,
    *,
    switchpoints: Optional[Union[Sequence[float], Dict[float, str]]] = None,
    mcmc_samples: int = 0,
    mu_prior: float = 0,
    drift_scale: float,
    nu: float = 5,
    x0_prior: float = 0.25,
    student_t: Optional[bool] = None,
    switchpoint_prob: float = 0.01,
    replicate_id: str = "unnamed",
):
    """Models a vector of growth rates to describe the observations.

    A MAP estimate is automatically determined.

    Parameters
    ----------
    t : numpy.ndarray
        Vector of data timepoints
    y : numpy.ndarray
        backscatter vector
    calibration_model : calibr8.CalibrationModel
        A calibration model for the CDW/observation relationship.
    switchpoints : optional, array-like or dict
        switchpoint times in the model (treated as inclusive upper bounds if they match a timepoint)
        if specified as a dict, the keys must be the timepoints
    mcmc_samples : int
        number of posterior draws (default to 0)
        This kwarg is a shortcut to run `result.sample(draw=mcmc_samples)`.
    mu_prior : float
        Prior belief in the growth rate at the beginning.
        Defaults to 0 which works well if there was a lag phase.
    drift_scale : float
        Standard deviation or scale of the random walk (how much µ_t drifts per timestep).
        This controls the bias-variance tradeoff of the method.
    nu : float
        Degree of freedom for StudentT random walks.
        This controls the prior probability of switchpoints.
    x0_prior : float
        prior expectation of initial biomass [g/L]
    student_t : optional, bool
        switches between a Gaussian or StudentT random walk
        if not set, it defaults to the expression `len(switchpoints) == 0`
    switchpoint_prob : float
        Probability level for automatic switchpoint detection (when `student_t=True`).
        Growth rate segments that lie outside of the (1 - switchpoint_prob) * 100 %
        equal tailed prior probability interval are classified as switchpoints.
    replicate_id : str
        name of the replicate that the data belongs to (defaults to "unnamed")

    Returns
    -------
    result : GrowthRateResult
        wraps around the data, model and fitting results
    """
    if not isinstance(switchpoints, dict):
        switchpoints = {t_switch: "known" for t_switch in switchpoints or []}
    t_switchpoints_known = numpy.sort(list(switchpoints.keys()))
    if student_t is None:
        student_t = len(switchpoints) == 0

    # Use a smoothed, diff-based growth rate on the backscatter to initialize the optimization.
    # These values are still everything but high-quality estimates of the growth rate,
    # but this intialization makes the optimization much more reliable.
    mu_guess = _get_smoothed_mu(t, y, calibration_model)

    t_data = t
    t_segments = numpy.mean([t_data[1:], t_data[:-1]], axis=0)
    TD = len(t_data)
    TS = len(t_segments)

    # build PyMC model
    coords = {
        "timepoint": numpy.arange(TD),
        "segment": numpy.arange(TS),
    }
    with pm.Model(coords=coords) as pmodel:
        pm.ConstantData("known_switchpoints", t_switchpoints_known)
        pm.ConstantData("t_data", t_data, dims="timepoint")
        pm.ConstantData("t_segments", t_segments, dims="segment")
        dt = pm.ConstantData("dt", numpy.diff(t_data), dims="segment")

        # The init dist for the random walk is where each segment starts.
        # Here we center it on the user-provided mu_prior,
        # taking the absolute of it (+0.05 safety margin to avoid 0) as the scale.
        init_dist = pm.Normal.dist(mu=mu_prior, sigma=pt.abs(mu_prior) + 0.05)

        if len(t_switchpoints_known) > 0:
            _log.info(
                "Creating model with %d switchpoints. StudentT=%b", len(t_switchpoints_known), student_t
            )
            # the growth rate vector is fragmented according to t_switchpoints_known
            mu_segments = []
            i_from = 0
            for i, t_switch in enumerate(t_switchpoints_known):
                i_to = int(numpy.argmax(t > t_switch))
                i_len = len(t[i_from:i_to])
                name = f"mu_phase_{i}"
                slc = slice(i_from, i_to)
                mu_segments.append(
                    _make_random_walk(
                        name,
                        init_dist=init_dist,
                        mu=0,
                        sigma=drift_scale,
                        nu=nu,
                        length=i_len,
                        student_t=student_t,
                        initval=mu_guess[slc],
                    )
                )
                i_from += i_len
            # the last segment until the end
            i_len = len(t[i_from:]) - 1
            name = f"mu_phase_{len(mu_segments)}"
            slc = slice(i_from, None)
            mu_segments.append(
                _make_random_walk(
                    name,
                    init_dist=init_dist,
                    mu=0,
                    sigma=drift_scale,
                    nu=nu,
                    length=i_len,
                    student_t=student_t,
                    initval=mu_guess[slc],
                )
            )
            mu_t = pm.Deterministic("mu_t", pt.concatenate(mu_segments), dims="segment")
        else:
            _log.info(
                "Creating model without switchpoints. StudentT=%b", len(t_switchpoints_known), student_t
            )
            mu_t = _make_random_walk(
                "mu_t",
                init_dist=init_dist,
                mu=0,
                sigma=drift_scale,
                nu=nu,
                length=TS,
                student_t=student_t,
                initval=mu_guess,
                dims="segment",
            )

        X0 = pm.LogNormal("X0", mu=numpy.log(x0_prior), sigma=1)
        Xt = pm.Deterministic(
            "X",
            pt.concatenate([X0[None], X0 * pm.math.exp(pt.extra_ops.cumsum(mu_t * dt))]),
            dims="timepoint",
        )
        calibration_model.loglikelihood(
            x=Xt,
            y=pm.ConstantData("backscatter", y, dims=("timepoint",)),
            name=f"{replicate_id}_{calibration_model.dependent_key}",
            dims="timepoint",
        )

    # MAP fit
    with pmodel:
        theta_map = pm.find_MAP(maxeval=15_000)

    # with StudentT random walks, switchpoints can be autodetected
    if student_t:
        switchpoints_detected = detect_switchpoints(
            switchpoint_prob,
            t_data,
            pmodel,
            theta_map,
        )
        # Known switchpoints override detected ones 👇
        switchpoints = {**switchpoints_detected, **switchpoints}

    # bundle up all relevant variables into a result object
    result = GrowthRateResult(
        t_data=t_data,
        t_segments=t_segments,
        y=y,
        calibration_model=calibration_model,
        switchpoints=switchpoints,
        pmodel=pmodel,
        theta_map=theta_map,
    )
    if len(result.detected_switchpoints):
        _log.info(
            "Detected %d previously unknown switchpoints: %s",
            len(result.detected_switchpoints),
            result.detected_switchpoints,
        )
    if mcmc_samples > 0:
        result.sample(draws=mcmc_samples)

    return result


def detect_switchpoints(
    switchpoint_prob: float,
    t_data: Sequence[float],
    pmodel: pm.Model,
    theta_map: Dict[str, numpy.ndarray],
) -> Dict[float, str]:
    """Helper function to detect switchpoints from a fitted random walk.

    Parameters
    ----------
    switchpoint_prob
        Probability threshold for detecting switchpoints.
        Random walk innovations with a prior probability less than this
        will be classified as switchpoints.
    t_data
        Time values corresponding to the random walk steps.
    pmodel
        The PyMC model containing `"mu_t*"` random walks.
    theta_map
        MAP estimate of the model.

    Returns
    -------
    switchpoints
        Dictionary of switchpoints with
        keys being the time point and
        values `"detected"`.
    """
    # first CDF values at all mu_t elements
    cdf_evals = []
    for rvname in sorted(theta_map.keys()):
        if rvname not in pmodel.named_vars:
            continue
        # The random walk may be split in multiple segments.
        # We can identify a segment from the RVOp type that created it.
        rv = pmodel[rvname]
        if rv.owner is None:
            continue
        if isinstance(rv.owner.op, pm.RandomWalk.rv_type):
            # Get a handle on the innovation dist so we can evaluate prior CDFs.
            innov_dist = rv.owner.inputs[1]
            # Calculate the innovations from the MAP estimate of the points.
            # This gives only the deltas between the points, so the 0th element
            # in the new vector corresponds to the segment between the 0st and 1nd point.
            innov = numpy.diff(theta_map[rvname])
            # Now we can evaluate the CDFs  of the innovations.
            logcdfs = pm.logcdf(innov_dist, innov).eval()
            # We define switchpoints based on the time of the point with an extreme CDF value.
            # To get our <number of segments> length vector to align with the <number of points>,
            # we prepend a 0.5 as a placeholder for the CDF of the initial point of the random walk.
            cdf_evals += [0.5, *numpy.exp(logcdfs)]
    cdf_evals_arr = numpy.array(cdf_evals)
    if len(cdf_evals_arr) != len(t_data) - 1:
        raise Exception(
            f"Failed to find all random walk segments. Found {len(cdf_evals_arr)}, expected {len(t_data) - 1}."
        )
    # Filter for the elements that lie outside of the [0.005, 0.995] interval (if switchpoint_prob=0.01).
    significance_mask = numpy.logical_or(
        cdf_evals_arr < (switchpoint_prob / 2),
        cdf_evals_arr > (1 - switchpoint_prob / 2),
    )
    # Collect switchpoint information from points with significant CDF values.
    # Here we don't need to filter known switchpoints, because these correspond to the first
    # point in each random walk, for which we assigned non-significant 0.5 CDF placeholders above.
    switchpoints = {t: "detected" for t, is_switchpoint in zip(t_data, significance_mask) if is_switchpoint}
    return switchpoints
