from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax.scipy import linalg
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNStandard, FunctionalModel
from parsmooth._base import are_inputs_compatible
from parsmooth._utils import none_or_shift, none_or_concat, mvn_loglikelihood
from parsmooth.linearization._cubature import _get_sigma_points
from parsmooth.linearization._common import get_mvnsqrt
from parsmooth.linearization import extended


def breakpoint_if_nonfinite(x, predict_ref):
    is_finite = jnp.isfinite(x.mean).all()

    def true_fn(x, predict_ref):
        pass

    def false_fn(x, predict_ref):
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x, predict_ref)


def sigma_point_approx(f: Callable, x: MVNStandard):
    """
    Approximate the expectation of a function with sigma points.
    :param f: The function to approximate.
    :param x: The Gaussian distribution to approximate the expectation of f with.
    :return: Approximation of the function's expectation.
    """
    x_sqrt = get_mvnsqrt(x)
    x_pts = _get_sigma_points(x_sqrt)

    f_pts = jax.vmap(f)(x_pts.points)
    m_f = jnp.tensordot(x_pts.wm, f_pts, 1)

    return m_f


def _mean_dynamics(model: FunctionalModel,
                   x: MVNStandard):
    f, _ = model
    return sigma_point_approx(f, x)


def _cov_dynamics(model: FunctionalModel,
                  x: MVNStandard):
    f, q = model
    F_x = jax.jacfwd(f)
    exp_F_x = sigma_point_approx(F_x, x)

    P = x.cov
    tmp = P @ exp_F_x.T
    return tmp + tmp.T + q.cov, tmp


def _predict(model: FunctionalModel,
            x: MVNStandard,
            dt: float):
    """Algorithm 9.3, @Sarkka2019."""
    pred_mean = x.mean + _mean_dynamics(model, x) * dt
    dP_k, dC_k = _cov_dynamics(model, x)
    pred_cov = x.cov + dP_k * dt
    C_k = x.cov + dC_k * dt
    gain = linalg.solve(pred_cov, C_k.T, assume_a='pos').T
    return MVNStandard(pred_mean, pred_cov), gain


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              dt: float,
              nominal_trajectory: Optional[MVNStandard] = None):
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    def body(carry, inp):
        x, ell = carry
        y, predict_ref, update_ref = inp

        if predict_ref is None:
            predict_ref = x
        x, gain = _predict(transition_model, predict_ref, dt)
        # breakpoint_if_nonfinite(x, predict_ref)
        if update_ref is None:
            update_ref = x
        H_x, cov_or_chol_R, c = extended(observation_model, update_ref)
        x, ell_inc = _standard_update(H_x, cov_or_chol_R, c, x, y)
        return (x, ell + ell_inc), (x, update_ref, gain)

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    (_, ell), (xs, predict_traj, gains) = jax.lax.scan(body, (x0, 0.), (observations, predict_traj, update_traj))
    xs = none_or_concat(xs, x0, 1)
    return xs, ell, predict_traj, gains


def smoothing(filtered_trajectory, predicted_trajectory, gains):
    last_state = jax.tree_map(lambda z: z[-1], filtered_trajectory)
    
    def body(smoothed, inp):
        filtered, predicted, gain = inp
        mean = filtered.mean + gain @ (smoothed.mean - predicted.mean)
        cov = filtered.cov + gain @ (smoothed.cov - predicted.cov) @ gain.T
        smoothed_state = MVNStandard(mean, cov)
        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_state,
                                      [none_or_shift(filtered_trajectory, -1), predicted_trajectory, gains],
                                      reverse=True)

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


def _standard_update(H, R, c, x, y):
    m, P = x

    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T
    chol_S = jnp.linalg.cholesky(S)
    G = P @ cho_solve((chol_S, True), H).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T
    ell = mvn_loglikelihood(y_diff, chol_S)
    return MVNStandard(m, P), ell
