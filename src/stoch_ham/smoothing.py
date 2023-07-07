from typing import Callable

import jax
import jax.scipy.linalg as jlag

from stoch_ham.base import MVNStandard, FunctionalModel
from stoch_ham.utils import none_or_shift, none_or_concat


def smoothing(transition_model: FunctionalModel,
              filter_trajectory: MVNStandard,
              linearization_method: Callable):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory)

    def body(smoothed, filtered):
        F_x, cov_or_chol, b = linearization_method(transition_model, smoothed)
        smoothed_state = _standard_smooth(F_x, cov_or_chol, b, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_state,
                                      none_or_shift(filter_trajectory, -1),
                                      reverse=True)

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


def _standard_smooth(F, Q, b, xf, xs):
    mf, Pf = xf
    ms, Ps = xs

    mean_diff = ms - (b + F @ mf)
    S = F @ Pf @ F.T + Q
    cov_diff = Ps - S

    gain = Pf @ jlag.solve(S, F, sym_pos=True).T
    ms = mf + gain @ mean_diff
    Ps = Pf + gain @ cov_diff @ gain.T

    return MVNStandard(ms, Ps)
