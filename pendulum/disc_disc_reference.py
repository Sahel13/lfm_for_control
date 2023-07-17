import jax
import jax.numpy as jnp
import jax.random as random

from parsmooth._base import MVNStandard, FunctionalModel
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing
from data import add_meas_noise

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds

########################################
# Get the data
########################################
seed = 2
key = random.PRNGKey(seed)

meas_error = jnp.array([.5, 2.5, 0.])

true_params = {
    'mass': 1.,
    'length': 2.,
    'q': 0.0001,
    'a': 3.,
    'omega': jnp.pi / 2
}

sampling_rate = 10
dt = 1./sampling_rate

x0_mean = jnp.array([1.5, 0., 0.])
t_span = (0., 10.)
ts = jnp.linspace(*t_span, int((t_span[1] - t_span[0])) * sampling_rate + 1)


def data_drift_fn(t, x, params):
    q, p, u = x
    g = 9.81
    m, l = params['mass'], params['length']
    du = params['a'] * params['omega'] * jnp.cos(params['omega'] * t)
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) + u, du])


sol = solve_ivp(data_drift_fn, t_span, x0_mean, t_eval=ts, args=(true_params,))
true_traj = sol.y.T
key, subkey = random.split(key)
observations = add_meas_noise(subkey, true_traj, meas_error)
observations = observations[:, :2]

plt.figure()
plt.plot(ts[1:], observations[:, 0], '.', label=r"$q$ (measured)")
plt.plot(ts[1:], observations[:, 1], '.', label=r"$p$ (measured)")
plt.plot(ts, true_traj[:, 0], label=r"$q$")
plt.plot(ts, true_traj[:, 1], label=r"$p$")
plt.plot(ts, true_traj[:, 2], label=r"$u$")
plt.title("Data")
plt.xlabel("Time")
plt.legend()
plt.show()

########################################
# Smoothing
########################################
def drift_fun(x, params):
    """
    The drift function of the augmented state.
    """
    q, p, u = x
    g = 9.81
    m, l, lamba = params[:3]
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) + u, -lamba * u])


def get_Q(params, dt):
    """
    Get the process noise covariance matrix `Q`
    by first defining the diffusion vector `L`.
    """
    res = 1e-6  # To prevent noise covariance from becoming singular.
    Q = jnp.diag(jnp.array([res, res, jnp.exp(params[-1])]))
    return Q * dt


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([1.5, 0., 0.])
    u0_cov = jnp.exp(params[-1]) / (2 * params[2])
    x0cov = jnp.diag(jnp.array([1., 1., u0_cov]))
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(params, observations, dt, meas_error):
    """
    Wrapper function to get the marginal data log-likelihood
    and the smoothed states.
    """
    # Define the transition model.
    Q = get_Q(params, dt)
    transition_model = FunctionalModel(
        lambda x: x + drift_fun(x, params) * dt,
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
    smoothed_states, ell = iterated_smoothing(
        observations, x0, transition_model, observation_model, extended,
        return_loglikelihood=True
    )

    return ell, smoothed_states


########################################
# Parameter estimation
########################################
# Define helper methods for optimization.
get_neg_log_lik = lambda params: -get_ell_and_filter(params, observations, dt, meas_error)[0]
grad_log_lik = jax.jit(jax.value_and_grad(get_neg_log_lik))


# Using L-BFGS-B
def wrapper_func(params):
    loss, grad_val = grad_log_lik(params)
    return np.array(loss, dtype=np.float64), np.array(grad_val, dtype=np.float64)


# Set up and run L-BFGS-B.
guess_params = np.array([1.5, 1., 1., 0.1])
bounds = Bounds([0.5, 0.5, 1e-2, -np.inf], [np.inf, np.inf, np.inf, np.inf])
opt_result = minimize(wrapper_func, guess_params, method='L-BFGS-B', jac=True, bounds=bounds)

# Visualize results.
best_params = opt_result.x
log_lik, smoothed_states = get_ell_and_filter(best_params, observations, dt, meas_error)
print(f"The best parameters are: {best_params} with log-likelihood {log_lik:.4f}.")

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
