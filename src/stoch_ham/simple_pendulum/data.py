import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from stoch_ham._sde import euler_maruyama


def hamiltonian(x, params):
    """
    The Hamiltonian of the simple pendulum.
    :param x: The state.
    :param params: The parameters of the system.
    :return: The Hamiltonian.
    """
    q, p = x
    m, l = params[:2]
    return p**2/(2 * m * l**2) + m * 9.81 * l * (1 - jnp.cos(q))


@jax.jit
def drift_fn(x, params):
    """
    The drift function of the stochastic Hamiltonian system.
    :param x: The state.
    :param params: The parameters of the system.
    :return: The drift vector field.
    """
    q, p = x
    g = 9.81
    m, l = params[:2]
    return jnp.array([p/(m * l**2), -m * g * l * jnp.sin(q)])


def diffusion_fn(x, params):
    """
    The diffusion function of the stochastic Hamiltonian system.
    :param x: The state.
    :param params: The parameters of the system.
    :return: The diffusion vector field.
    """
    return jnp.array([0., -1.])


seed = 123
key = jax.random.PRNGKey(seed)

param_dict = {
    'm': 1.,
    'l': 2.,
    'lambda': 5.,
    'Q': 1.
}
params = jnp.array([val for val in param_dict.values()])

x0 = jnp.array([jnp.pi/2, 0.])
t_span = (0., 40.)
dt = 0.001

soln = euler_maruyama(key, drift_fn, diffusion_fn, params, x0, t_span, dt)
ts = jnp.linspace(*t_span, len(soln))

plt.figure()
plt.plot(soln[:, 0], soln[:, 1])
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
plt.title("Phase space trajectory")
plt.show()

plt.figure()
plt.plot(ts, soln[:, 0], label=r"$q$")
plt.plot(ts, soln[:, 1], label=r"$p$")
plt.title("Trajectory")
plt.xlabel("Time")
plt.legend()
plt.show()

energies = jax.vmap(hamiltonian, in_axes=(0, None))(soln, params)
plt.figure()
plt.plot(ts, energies)
plt.title("Energy vs time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.show()
