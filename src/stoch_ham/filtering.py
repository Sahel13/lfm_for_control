from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.typing import ArrayLike

from stoch_ham.base import MVNStandard
from stoch_ham.utils import mvn_loglikelihood, none_or_concat


def filtering(observations: ArrayLike,
              x0: MVNStandard,
              transition_model: Callable,
              H: ArrayLike,
              R: ArrayLike,
              linearization_method: Callable
              ):
    def body(carry, y):
        x, ell = carry

        # Predict
        F_x, cov_Q, b = linearization_method(transition_model, x)
        x = predict(F_x, cov_Q, b, x)

        # Update (Linear observation model assumed)
        x, ell_inc = update(H, R, x, y)
        return (x, ell + ell_inc), x

    (_, ell), xs = jax.lax.scan(body, (x0, 0.), observations)
    xs = none_or_concat(xs, x0, 1)

    return xs, ell


def predict(F, Q, b, x: MVNStandard):
    m, P = x

    m = F @ m + b
    P = Q + F @ P @ F.T

    return MVNStandard(m, P)


def update(H, R, x: MVNStandard, y: ArrayLike):
    m, P = x

    y_hat = H @ m
    y_diff = y - y_hat
    S = R + H @ P @ H.T
    chol_S = jnp.linalg.cholesky(S)
    G = P @ cho_solve((chol_S, True), H).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T

    # Get the marginal data log-likelihood.
    ell = mvn_loglikelihood(y, y_hat, chol_S)
    return MVNStandard(m, P), ell
