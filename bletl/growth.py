import logging
import numpy
import scipy.stats
import typing

import arviz
import pymc3
import theano.tensor as tt

import calibr8

from . import splines

_log = logging.getLogger(__file__)


class GrowthRateResult:
    """ Represents the result of applying the µ(t) model to one dataset. """
    def __init__(
        self,
        *,
        t:numpy.ndarray,
        y:numpy.ndarray,
        calibration_model: calibr8.CalibrationModel,
        switchpoints:typing.Dict[float, str],
        pmodel:pymc3.Model,
        theta_map:dict,
    ):
        """ Creates a result object of a growth rate analysis.

        Parameters
        ----------
        t : numpy.ndarray
            time vector
        y : numpy.ndarray
            backscatter vector
        switchpoints : dict
            maps switchpoint times to labels
        pmodel : pymc3.Model
            the PyMC3 model underlying the analysis
        theta_map : dict
            the PyMC3 MAP estimate
        """
        self._t = t
        self._y = y
        self._switchpoints = switchpoints
        self.calibration_model = calibration_model
        self._pmodel = pmodel
        self._theta_map = theta_map
        self._idata = None
        super().__init__()

    @property
    def t(self) -> numpy.ndarray:
        """ Vector of data timepoints. """
        return self._t
    
    @property
    def y(self) -> numpy.ndarray:
        """ Vector of backscatter observations. """
        return self._y

    @property
    def switchpoints(self) -> typing.Dict[float, str]:
        """ Dictionary (by time) of known and detected switchpoints. """
        return self._switchpoints

    @property
    def known_switchpoints(self) -> typing.Tuple[float]:
        """ Time values of previously known switchpoints in the model. """
        return tuple(
            t
            for t, label in self.switchpoints.items()
            if label != 'autodetected'
        )

    @property
    def detected_switchpoints(self) -> typing.Tuple[float]:
        """ Time values of switchpoints that were autodetected from the fit. """
        return tuple(
            t
            for t, label in self.switchpoints.items()
            if label == 'automatic'
        )

    @property
    def pmodel(self) -> pymc3.Model:
        """ The PyMC3 model underlying this analysis. """
        return self._pmodel

    @property
    def theta_map(self) -> dict:
        """ MAP estimate of the model parameters. """
        return self._theta_map

    @property
    def idata(self) -> typing.Optional[arviz.InferenceData]:
        """ ArviZ InferenceData object of the MCMC trace. """
        return self._idata

    @property
    def mu_map(self) -> numpy.ndarray:
        """ MAP estimate of the growth rates. """
        return self.theta_map['mu_t']

    @property
    def x_map(self) -> numpy.ndarray:
        """ MAP estimate of the biomass curve. """
        return self.theta_map['X']

    @property
    def mu_mcmc(self) -> typing.Optional[numpy.ndarray]:
        """ Posterior samples of growth rates. """
        if not self.idata:
            return None
        return self.idata.posterior.mu_t.stack(sample=('chain', 'draw')).values.T

    @property
    def x_mcmc(self) -> typing.Optional[numpy.ndarray]:
        """ Posterior samples of biomass curve. """
        if not self.idata:
            return None
        return self._idata.posterior['X'].stack(sample=('chain', 'draw')).T

    def sample(self, **kwargs) -> None:
        """ Runs MCMC sampling with default settings on the growth model.

        Parameters
        ----------
        **sample_kwargs
            optional keyword-arguments to pymc3.sample(...) to override defaults
        """
        sample_kwargs = dict(
            return_inferencedata=True,
            target_accept=0.95,
            init='jitter+adapt_diag',
            start=self.theta_map,
            tune=500,
            draws=500,
        )
        sample_kwargs.update(kwargs)
        with self.pmodel:
            self._idata = pymc3.sample(**sample_kwargs)
        return


def _make_random_walk(name:str, *, sigma:float, nu: float=1, length:int, student_t:bool, initval: numpy.ndarray=None):
    """ Create a random walk with either a Normal or Student-t distribution.

    Parameteres
    -----------
    name : str
        Name of the random walk variable.
    sigma : float
        Standard deviation (Normal) or scale (StudentT) parameter.
    nu : float
        Degree of freedom for the StudentT distribution - only used when `student_t == True`.
    length : int
        Number of steps in the random walk.
    student_t : bool
        If `True` a `pymc3.Deterministic` of a StudentT-random walk is created.
        Otherwise a `GaussianRandomWalk` is created.
    initval : numpy.ndarray
        Initial values for the RandomWalk variable.
        If set, PyMC3 uses these values as start points for MAP optimization and MCMC sampling.

    Returns
    -------
    random_walk : TensorVariable
        The tensor variable of the random walk.
    """
    if student_t:
        # a random walk of length N is just the cumulative sum over a N-dimensional random variable:
        return pymc3.Deterministic(name, tt.cumsum(
            pymc3.StudentT(f'{name}__diff_', mu=0, sd=sigma, nu=5, shape=(length,), testval=numpy.diff(initval, prepend=0))
        ))
    else:
        return pymc3.GaussianRandomWalk(name, mu=0, sigma=sigma, shape=(length,), testval=initval)


def _get_smoothed_mu(t: numpy.ndarray, y: numpy.ndarray, cm_cdw: calibr8.CalibrationModel, *, clip=0.8) -> numpy.ndarray:
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
    y = numpy.convolve(y, numpy.ones(5)/5, "same")
    
    # convert to biomass
    X = cm_cdw.predict_independent(y)
    
    # calculate growth rate
    dX = numpy.diff(X, prepend=numpy.median(X[0:5]))
    dt = numpy.diff(t, prepend=0)
    mu = (dX / dt) / X
    
    # clip growth rate into a realistic interval
    mu = numpy.clip(mu, -clip, clip)
    
    # smooth again to reduce peaking
    mu = numpy.convolve(mu, numpy.ones(5)/5, "same")
    return mu


def fit_mu_t(
    t:typing.Sequence[float],
    y:typing.Sequence[float],
    calibration_model:calibr8.CalibrationModel,
    *,
    switchpoints:typing.Optional[typing.Union[typing.Sequence[float], typing.Dict[float, str]]]=None,
    mcmc_samples:int=0,
    σ:float=0.01,
    x0_prior:float=0.25,
    student_t:typing.Optional[bool]=None,
    replicate_id:str='unnamed'
):
    """ Models a vector of growth rates to describe the observations.

    A MAP estimate is automatically determined.

    Parameters
    ----------
    t : numpy.ndarray
        time vector
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
    σ : float
        standard deviation of the random walk - this controls how fast the growth rate may drift
    x0_prior : float
        prior expectation of initial biomass [g/L]
    student_t : optional, bool
        switches between a Gaussian or StudentT random walk
        if not set, it defaults to the expression `len(switchpoints) == 0`
    replicate_id : str
        name of the replicate that the data belongs to (defaults to "unnamed")

    Returns
    -------
    result : GrowthRateResult
        wraps around the data, model and fitting results
    """
    if not isinstance(switchpoints, dict):
        switchpoints = {
            t_switch : 'known'
            for t_switch in switchpoints or []
        }
    t_switchpoints_known = numpy.sort(list(switchpoints.keys()))
    if student_t is None:
        student_t = len(switchpoints) == 0
    # build a dict of known switchpoint begin cycle indices so they can be ignored in autodetection
    c_switchpoints_known = [0]

    # Use a smoothed, diff-based growth rate on the backscatter to initialize the optimization.
    # These values are still everything but high-quality estimates of the growth rate,
    # but this intialization makes the optimization much more reliable.
    mu_guess = _get_smoothed_mu(t, y, calibration_model)

    # build PyMC3 model
    coords = {
        'time': t
    }
    with pymc3.Model(coords=coords) as pmodel:
        pymc3.Data('known_switchpoints', t_switchpoints_known)
        dt = pymc3.Data('dt', numpy.diff(t, prepend=0), dims='time')
    
        if len(t_switchpoints_known) > 0:
            _log.info('Creating model with %d switchpoints. StudentT=%b', len(t_switchpoints_known), student_t)
            # the growth rate vector is fragmented according to t_switchpoints_known
            mu_segments = []
            i_from = 0
            for i, t_switch in enumerate(t_switchpoints_known):
                i_to = numpy.argmax(t > t_switch)
                i_len = len(t[i_from:i_to])
                name = f'mu_phase_{i}'
                mu_segments.append(
                    _make_random_walk(name, sigma=σ, length=i_len, student_t=student_t, initval=mu_guess[i_from:i_to])
                )
                i_from += i_len
                # remember the index to ignore it in potential autodetection
                c_switchpoints_known.append(i_from)
            # the last segment until the end
            i_len = len(t[i_from:])
            name = f'mu_phase_{len(mu_segments)}'
            mu_segments.append(
                _make_random_walk(name, sigma=σ, length=i_len, student_t=student_t, initval=mu_guess[i_from:])
            )
            mu_t = pymc3.Deterministic('mu_t', tt.concatenate(mu_segments))
        else:
            _log.info('Creating model without switchpoints. StudentT=%b', len(t_switchpoints_known), student_t)
            mu_t = _make_random_walk('mu_t', sigma=σ, length=len(t), student_t=student_t)
    
        X0 = pymc3.Lognormal('X0', mu=numpy.log(x0_prior), sd=1)
        Xt = pymc3.Deterministic('X', X0 + X0 * pymc3.math.exp(tt.extra_ops.cumsum(mu_t * dt)))
        calibration_model.loglikelihood(
            x=Xt,
            y=pymc3.Data('backscatter', y, dims=('time',)),
            replicate_id=replicate_id,
            dependent_key=calibration_model.dependent_key
        )

    # MAP fit
    with pmodel:
        theta_map = pymc3.find_MAP(maxeval=15_000)

    # with StudentT random walks, switchpoints can be autodetected
    if student_t:
        # first CDF values at all mu_t elements
        cdf_evals = []
        for rvname in sorted(theta_map.keys()):
            if '__diff_' in rvname:
                rv = pmodel[rvname]
                # for every µ, find out where it lies in the CDF of the StudentT prior distribution
                cdf_evals += list(scipy.stats.t.cdf(
                    x=theta_map[rvname],
                    loc=rv.distribution.mu.eval(),
                    scale=rv.distribution.sd.eval(),
                    df=rv.distribution.nu.eval(),
                ))
        cdf_evals = numpy.array(cdf_evals)
        # filter for the elements that lie outside of the [0.005, 0.995] interval
        significance_mask = numpy.logical_or(
            cdf_evals < 0.005,
            cdf_evals > 0.995,
        )
        # add these autodetected timepoints to the switchpoints-dict
        # (ignore the first timepoint)
        for c_switch, (t_switch, is_switchpoint) in enumerate(zip(numpy.insert(t, 0, 0), significance_mask)):
            if is_switchpoint and c_switch not in c_switchpoints_known:
                switchpoints[t_switch] = 'detected'
    
    # bundle up all relevant variables into a result object
    result = GrowthRateResult(
        t=t, y=y,
        calibration_model=calibration_model,
        switchpoints=switchpoints,
        pmodel=pmodel,
        theta_map=theta_map,
    )
    if len(result.detected_switchpoints):
        _log.info('Detected %d previously unknown switchpoints: %s', len(result.detected_switchpoints), result.detected_switchpoints)
    if mcmc_samples > 0:
        result.sample(draws=mcmc_samples)           

    return result
