# Test whether the EKF code works.
import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

from stoch_ham.base import MVNStandard, FunctionalModel
from stoch_ham.filtering import filtering
from stoch_ham.linearization import extended

# Get the data
t_span = (0., 10.)
num_points = 100
t_eval = jnp.linspace(t_span[0], t_span[1], num_points + 1)
x_0 = MVNStandard(jnp.array([1., 0.]), jnp.identity(2))

key = random.PRNGKey(10)

def dyn_fun(x):
    return jnp.sin(x)

H = jnp.identity(2)
def meas_fun(x):
    return H @ x


Q, R = 0.1 * jnp.identity(2), 0.05 * jnp.identity(2)


def get_data(rng_key):
    gen_key, use_key = random.split(rng_key)
    process_noise = random.normal(use_key, shape=(num_points, 2)) @ jnp.linalg.cholesky(Q)
    gen_key, use_key = random.split(rng_key)
    obs_noise = random.normal(use_key, shape=(num_points, 2)) @ jnp.linalg.cholesky(R)
    true_traj = []
    x = random.multivariate_normal(gen_key, mean=x_0.mean, cov=x_0.cov)
    true_traj.append(x)
    for i in range(num_points):
        x = dyn_fun(x) + process_noise[i]
        true_traj.append(x)

    true_traj = jnp.vstack(true_traj)
    meas = true_traj[1:] + obs_noise

    return true_traj, meas


key, subkey = random.split(key)
true_traj, meas = get_data(subkey)

plt.figure()
plt.plot(t_eval, true_traj[:, 0], label='True trajectory')
plt.plot(t_eval[1:], meas[:, 0], '.', label='Measurements')
plt.legend()
plt.show()

# EKF
transition_model = FunctionalModel(dyn_fun, MVNStandard(0, Q))
observation_model = FunctionalModel(meas_fun, MVNStandard(0, R))
filt_states, log_likelihood = filtering(meas, x_0, transition_model, observation_model, extended)

plt.figure()
plt.plot(t_eval, true_traj[:, 0], label='True trajectory')
plt.plot(t_eval[1:], meas[:, 0], '.', label='Measurements')
plt.plot(t_eval, filt_states.mean[:, 0], label='Filtered mean')
plt.legend()
plt.show()
