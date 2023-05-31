from typing import Callable

import jax
from jax.typing import ArrayLike

from stoch_ham.base import MVNStandard, FunctionalModel


def extended(model: FunctionalModel, x: MVNStandard):
    f, q = model
    m_x, _ = x
    return _standard_linearize_callable(f, m_x, *q)


def _linearize_callable_common(f: Callable, x: ArrayLike):
    return f(x), jax.jacfwd(f, 0)(x)


def _standard_linearize_callable(f: Callable, x: ArrayLike, m_q: ArrayLike, cov_q: ArrayLike):
    res, F_x = _linearize_callable_common(f, x)
    return F_x, cov_q, res - F_x @ x + m_q
