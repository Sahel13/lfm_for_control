import jax
import jax.numpy as jnp
import jax.random as random

from stoch_ham.simple_pendulum.data import get_dataset, hamiltonian
from stoch_ham.filtering import filtering
from stoch_ham.linearization import extended
from stoch_ham.base import MVNStandard, FunctionalModel

import matplotlib.pyplot as plt

####################
# Get the data
####################
seed = 12
key = random.PRNGKey(seed)

true_params = {
    'mass': 1.,
    'length': 2.,
    'lambda': 5.,
    'q': 0.05
}

x0_mean = jnp.array([jnp.pi / 2, 0.])
t_span = (0., 10.)

true_traj, observations = get_dataset(key, 1, true_params, x0_mean, t_span)[0]

ts = jnp.linspace(*t_span, len(observations))

plt.figure()
plt.plot(observations[:, 0], observations[:, 1])
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
plt.title("Phase space trajectory")
plt.show()

plt.figure()
plt.plot(ts, observations[:, 0], '.', label=r"$q$ (measured)")
plt.plot(ts, observations[:, 1], '.', label=r"$p$ (measured)")
plt.plot(ts, true_traj[:, 0], label=r"$q$")
plt.plot(ts, true_traj[:, 1], label=r"$p$")
plt.title("Trajectory")
plt.xlabel("Time")
plt.legend()
plt.show()

energies = jax.vmap(hamiltonian, in_axes=(0, None))(observations, true_params)
plt.figure()
plt.plot(ts, energies)
plt.title("Energy vs time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.show()


####################
# Filtering
####################
guess_params = {
    'mass': 2.,
    'length': 3.,
    'lambda': 4.,
    'q': 0.01,
    'meas_error': jnp.array([0.1, 0.1])
}


def drift_fun(x, params):
    q, p, u = x
    g = 9.81
    m, l, lamba = params['mass'], params['length'], params['lambda']
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) - u, -lamba * u])


def transition_fun(x, params, dt=0.1):
    return x + drift_fun(x, params) * dt


def Q(params, dt):
    L = jnp.array([0., 0., 1.])[:, None]
    return L @ L.T * params['q'] * dt


def get_ell_and_filter(params):
    dt = 0.1
    x0mean = jnp.array([jnp.pi / 2, 0., 0.])
    P0 = params['q'] / (2 * params['lambda'])
    x0cov = jnp.diag(jnp.array([1., 1., P0]))
    x0 = MVNStandard(x0mean, x0cov)
    transition_model = FunctionalModel(
        lambda x: transition_fun(x, params),
        MVNStandard(jnp.zeros(3), Q(params, dt))
    )
    R = jnp.diag(params['meas_error'])
    H = jnp.array([[1., 0., 0.], [0., 1., 0.]])
    observation_model = FunctionalModel(
        lambda x: H @ x,
        MVNStandard(jnp.zeros(2), R)
    )

    filt_states, ell = filtering(observations, x0, transition_model, observation_model, extended)
    return ell, filt_states


ell, filt_states = get_ell_and_filter(guess_params)
print(ell)
