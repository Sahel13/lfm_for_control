import chex
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def mvn_loglikelihood(x, mean, chol_cov):
    """
    Log-likelihood (log-pdf) of a multivariate normal distribution.
    :param x: Sample.
    :param mean: Mean of the distribution.
    :param chol_cov: Cholesky decomposition of the covariance matrix.
    :return: Log-likelihood (scalar).
    """
    dim = chol_cov.shape[0]
    y = solve_triangular(chol_cov, x - mean, lower=True)
    normalizing_constant = (
            jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant


def none_or_concat(x, y, position=1):
    """Method to concatenate two pytrees."""
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)


def l2_loss_single(prediction, target):
    """
    L2 loss between two 1-D arrays corresponding to a single data point.
    """
    chex.assert_equal_shape((prediction, target))
    return jnp.sum(jnp.square(prediction - target))


def l2_loss(predictions, targets):
    """
    L2 loss between predictions and targets.
    predictions.shape == (num_points, input_dims)
    :return: Mean L2 loss.
    """
    return jnp.mean(jax.vmap(l2_loss_single)(predictions, targets))
