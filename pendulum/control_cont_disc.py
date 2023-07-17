import jax
import jax.numpy as jnp
import jax.random as random

from parsmooth._base import MVNStandard, FunctionalModel
from pendulum.data import get_dataset
from stoch_ham.continuous_discrete import filtering, smoothing

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds

####################
# Get the data
####################
seed = 1
key = random.PRNGKey(seed)

meas_error = jnp.array([.5, 2.5, 0.])

true_params = {
    'mass': 1.,
    'length': 2.,
    'q': 0.0001,
    'a': 3.,
    'omega': jnp.pi / 2
}

sampling_rate = 40
dt = 1./sampling_rate

x0_mean = jnp.array([1.5, 0., 0.])
t_span = (0., 10.)


def data_drift_fn(x, t, params):
    q, p, u = x
    g = 9.81
    m, l = params['mass'], params['length']
    du = params['a'] * params['omega'] * jnp.cos(params['omega'] * t)
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) + u, du])


def data_diffusion_fn(x, t, params):
    return jnp.array([0., 0., 1.])


true_traj, observations = get_dataset(
    key, true_params, x0_mean, t_span, meas_error,
    drift_fn=data_drift_fn, diffusion_fn=data_diffusion_fn, sampling_rate=sampling_rate)

observations = observations[:, :2]

ts = jnp.linspace(*t_span, len(true_traj))
print("Data generated.")

# plt.figure()
# plt.plot(observations[:, 0], observations[:, 1])
# plt.xlabel(r"$q$")
# plt.ylabel(r"$p$")
# plt.title("Phase space trajectory")
# plt.show()
#
# plt.figure()
# plt.plot(ts[1:], observations[:, 0], '.', label=r"$q$ (measured)")
# plt.plot(ts[1:], observations[:, 1], '.', label=r"$p$ (measured)")
# plt.plot(ts, true_traj[:, 0], label=r"$q$")
# plt.plot(ts, true_traj[:, 1], label=r"$p$")
# plt.plot(ts, true_traj[:, 2], label=r"$u$")
# plt.title("Trajectory")
# plt.xlabel("Time")
# plt.legend()
# plt.show()
#
# energies = jax.vmap(hamiltonian, in_axes=(0, None))(observations, true_params)
# plt.figure()
# plt.plot(ts[1:], energies)
# plt.title("Energy vs time")
# plt.xlabel("Time")
# plt.ylabel("Energy")
# plt.show()


####################
# Filtering
####################
def drift_fun(x, params):
    """
    The drift function of the augmented state.
    """
    q, p, u = x
    g = 9.81
    m, l, lamba = params[:3]
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) + u, -lamba * u])


def get_LQL(params):
    """
    Get the process noise covariance matrix `Q`
    by first defining the diffusion vector `L`.
    """
    eps = 1e-4
    LQL = jnp.diag(jnp.array([eps, eps, jnp.exp(params[-1])]))
    return LQL


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([1.5, 0., 0.])
    u0_cov = jnp.exp(params[-1]) / (2 * params[2])
    x0cov = jnp.diag(jnp.array([1., 1., u0_cov]))
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(params, observations, dt, meas_error, smooth=False):
    """
    Wrapper function to get the marginal data log-likelihood
    and the filtered states.
    """
    # Define the transition model.
    Q = get_LQL(params)
    transition_model = FunctionalModel(
        lambda x: drift_fun(x, params),
        MVNStandard(jnp.zeros(3), Q)
    )

    # Define the observation model.
    R = jnp.diag(meas_error[:2])
    H = jnp.array([[1., 0., 0.], [0., 1., 0.]])
    observation_model = FunctionalModel(
        lambda x: H @ x,
        MVNStandard(jnp.zeros(2), R)
    )

    # Get the initial state distribution and run the filter.
    x0 = get_x0(params)
    filtered_states, log_lik, predicted_trajectory, gains = filtering(
        observations, x0, transition_model, observation_model, dt)

    if smooth:
        smoothed_states = smoothing(filtered_states, predicted_trajectory, gains)
        return log_lik, filtered_states, smoothed_states

    return log_lik, filtered_states


####################
# Parameter estimation
####################
get_neg_log_lik = lambda params: -get_ell_and_filter(params, observations, dt, meas_error)[0]
grad_log_lik = jax.jit(jax.value_and_grad(get_neg_log_lik))


# Using L-BFGS-B
def wrapper_func(params):
    loss, grad_val = grad_log_lik(params)
    return np.array(loss, dtype=np.float64), np.array(grad_val, dtype=np.float64)


guess_params = np.array([1.5, 1., 1., 0.1])
bounds = Bounds([0.5, 0.5, 1e-2, -np.inf], [np.inf, np.inf, np.inf, np.inf])
opt_result = minimize(wrapper_func, guess_params, method='L-BFGS-B', jac=True, bounds=bounds)
best_params = opt_result.x

log_lik, filt_states, smoothed_states = get_ell_and_filter(guess_params, observations, dt, meas_error, True)
print(f"The best parameters are: {best_params} with log-likelihood {log_lik:.4f}.")
smoothed_states = filt_states

fig, axs = plt.subplots(3, 1, sharex=True, layout="tight")
axs[0].plot(ts, true_traj[:, 0], label="True")
axs[0].scatter(ts[1:], observations[:, 0], label="Measured")
axs[0].plot(ts, smoothed_states.mean[:, 0], label="Smoothed")
axs[0].set_ylabel(r"$q$")
axs[0].legend()

axs[1].plot(ts, true_traj[:, 1], label="True")
axs[1].scatter(ts[1:], observations[:, 1], label="Measured")
axs[1].plot(ts, smoothed_states.mean[:, 1], label="Smoothed")
axs[1].set_ylabel(r"$p$")
axs[1].legend()

axs[2].plot(ts, true_traj[:, 2], label="True")
axs[2].plot(ts, smoothed_states.mean[:, 2], label="Smoothed")
axs[2].set_ylabel(r"$u$")
axs[2].set_xlabel("Time t")
axs[2].legend()

fig.suptitle("Smoothed trajectory")
plt.show()
