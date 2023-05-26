import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(0, 1))
def euler_maruyama_step(f, L, params, x, dt, dbeta):
    """
    Single step of the Euler-Maruyama method for stochastic differential equations.
    :param f: The drift function.
    :param L: The diffusion function.
    :param params: The parameters of the system.
    :param x: The state at time t.
    :param dt: Time step.
    :param dbeta: Brownian motion increment corresponding to dt.
    :return: The state at time t + dt.
    """
    return x + f(x, params) * dt + L(x, params) * dbeta


def euler_maruyama(key, f, L, params, x0, t_span, dt):
    """
    Euler-Maruyama method for stochastic differential equations.
    :param key: JAX random key.
    :param f: The drift function.
    :param L: The diffusion function.
    :param params: The parameters of the system.
    :param x0: Initial state.
    :param t_span: Tuple(start, end) of the time interval to integrate in.
    :param dt: Time step.
    :return: Array(time_steps, state_dim).
    """
    time_steps = int((t_span[1] - t_span[0]) / dt)
    q = params[-1]
    dbeta = jnp.sqrt(q * dt) * jax.random.normal(key, shape=(time_steps,))
    soln = [x0]
    x = x0
    for i in range(time_steps):
        x = euler_maruyama_step(f, L, params, x, dt, dbeta[i])
        soln.append(x)

    return jnp.vstack(soln)
