from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_solve, solve_triangular

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, are_inputs_compatible, ConditionalMomentsModel
from parsmooth._utils import tria, none_or_shift, none_or_concat, mvn_loglikelihood
from parsmooth.linearization import extended


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: FunctionalModel,
              observation_model: MVNStandard,
              nominal_trajectory: Optional[MVNStandard] = None,
              ) -> MVNStandard:
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    def body(carry, inp):
        x = carry
        y, predict_ref, update_ref = inp

        if predict_ref is None:
            predict_ref = x
        F_x, cov_or_chol_Q, b = extended(transition_model, predict_ref)
        x = predict(F_x, cov_or_chol_Q, b, x)

        x = update(x, observation_model)
        return x, x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    _, xs = jax.lax.scan(body, x0, (observations, predict_traj, update_traj))
    xs = none_or_concat(xs, x0, 1)

    return xs


def predict(F, Q, b, x):
    m, P = x

    m = F @ m + b
    P = Q + F @ P @ F.T

    return MVNStandard(m, P)


def update(x: MVNStandard, y: MVNStandard):
    gamma = x.cov + y.cov
    K = solve(gamma, x.cov)
    P = x.cov - x.cov @ K
    S = solve(x.cov, x.mean)
    m = P @ S
    return MVNStandard(m, P)
