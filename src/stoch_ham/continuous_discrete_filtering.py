from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy import linalg
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNStandard, FunctionalModel
from parsmooth._base import are_inputs_compatible
from parsmooth._utils import none_or_shift, none_or_concat, mvn_loglikelihood
from parsmooth.linearization._gh import _get_sigma_points
from parsmooth.linearization._common import get_mvnsqrt
from parsmooth.linearization import extended


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

        # Predict
        if predict_ref is None:
            predict_ref = x
        x, gain = _predict(transition_model, predict_ref, dt)

        # Update
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


def sigma_point_approx(f: Callable, x: MVNStandard) -> Array:
    """
    Approximate the expectation of a function with sigma points.
    :param f: The function to approximate.
    :param x: The Gaussian distribution to approximate the expectation of f with.
    :return: Approximation of the function's expectation.
    """
    x_sqrt = get_mvnsqrt(x)
    x_pts = _get_sigma_points(x_sqrt, order=3)

    f_pts = jax.vmap(f)(x_pts.points)
    m_f = jnp.tensordot(x_pts.wm, f_pts, 1)

    return m_f


def _mean_dynamics(model: FunctionalModel, x: MVNStandard) -> Array:
    f, _ = model
    return sigma_point_approx(f, x)


def _cov_dynamics(model: FunctionalModel, x: MVNStandard) -> Tuple[Array, Array]:
    f, q = model
    F_x = jax.jacfwd(f)
    exp_F_x = sigma_point_approx(F_x, x)

    P = x.cov
    tmp = P @ exp_F_x.T
    return tmp + tmp.T + q.cov, tmp


def _joint_dynamics(model: FunctionalModel, x: MVNStandard):
    dmdt = _mean_dynamics(model, x)
    dPdt, dCdt = _cov_dynamics(model, x)
    return MVNStandard(dmdt, dPdt), dCdt


def rk4_step(f: Callable, x: MVNStandard, dt: float):
    """
    An RK4 step for a function `f` that outputs an MVN distribution.
    """
    k1 = f(x)
    k2 = f(jax.tree_map(lambda a, b: a + dt / 2 * b, x, k1))
    k3 = f(jax.tree_map(lambda a, b: a + dt / 2 * b, x, k2))
    k4 = f(jax.tree_map(lambda a, b: a + dt * b, x, k3))
    return jax.tree_map(lambda a, b, c, d, e: a + dt/6 * (b + 2 * c + 2 * d + e), x, k1, k2, k3, k4)


def _predict(model: FunctionalModel,
             x: MVNStandard,
             dt: float):
    """
    Prediction step of the continuous-discrete filter.
    Algorithm 9.3, @Sarkka2019.
    """
    # Euler
    # dmdt = _mean_dynamics(model, x)
    # dPdt, dCdt = _cov_dynamics(model, x)
    # pred_mean = x.mean + dt * dmdt
    # pred_cov = x.cov + dt * dPdt
    # pred_x = MVNStandard(pred_mean, pred_cov)
    # RK4
    pred_x = rk4_step(lambda xi: _joint_dynamics(model, xi)[0], x, dt)
    _, dCdt = _cov_dynamics(model, x)
    pred_C = x.cov + dt * dCdt
    gain = linalg.solve(pred_x.cov, pred_C.T, assume_a='pos').T
    return pred_x, gain


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
