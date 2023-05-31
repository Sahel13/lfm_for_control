from jax.typing import ArrayLike
from typing import NamedTuple, Callable


class MVNStandard(NamedTuple):
    """Multivariate Normal Distribution with mean and covariance."""
    mean: ArrayLike
    cov: ArrayLike


class FunctionalModel(NamedTuple):
    function: Callable
    mvn: MVNStandard
