from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.typing import ArrayLike

from stoch_ham.base import MVNStandard, FunctionalModel
from stoch_ham.utils import mvn_loglikelihood, none_or_concat, none_or_shift


def filtering(observations: ArrayLike,
              x0: MVNStandard,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method: Callable,
              nominal_trajectory: Optional[MVNStandard] = None
              ):
    def body(carry, inp):
        x, log_lik = carry
        y, predict_ref, update_ref = inp

        # Predict
        if predict_ref is None:
            predict_ref = x
        F_x, cov_Q, b = linearization_method(transition_model, predict_ref)
        x = predict(F_x, cov_Q, b, x)

        # Update
        if update_ref is None:
            update_ref = x
        H_x, cov_R, c = linearization_method(observation_model, update_ref)
        x, log_lik_inc = update(H_x, cov_R, c, x, y)
        return (x, log_lik + log_lik_inc), x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    (_, log_lik), xs = jax.lax.scan(body, (x0, 0.), (observations, predict_traj, update_traj))
    xs = none_or_concat(xs, x0, 1)

    return xs, log_lik


def predict(F, Q, b, x: MVNStandard):
    m, P = x

    m = F @ m + b
    P = Q + F @ P @ F.T

    return MVNStandard(m, P)


def update(H, R, c, x: MVNStandard, y: ArrayLike):
    m, P = x

    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T
    chol_S = jnp.linalg.cholesky(S)
    G = P @ cho_solve((chol_S, True), H).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T

    # Get the marginal data log-likelihood.
    log_lik = mvn_loglikelihood(y, y_hat, chol_S)
    return MVNStandard(m, P), log_lik
