import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from stoch_ham.base import MVNStandard, FunctionalModel
from stoch_ham.filtering import filtering
from stoch_ham.smoothing import smoothing
from stoch_ham.linearization import extended
from stoch_ham.simple_pendulum.data import get_dataset, hamiltonian

import numpy as np
import optax
from scipy.optimize import minimize

####################
# Get the data
####################
seed = 1
key = random.PRNGKey(seed)

meas_error = jnp.array([.5, 2.5])

true_params = {
    'mass': 1.,
    'length': 2.,
    'q': 0.5
}

sampling_rate = 10
dt = 1./sampling_rate

x0_mean = jnp.array([1.5, 0.])
t_span = (0., 10.)

true_traj, observations = get_dataset(
    key, true_params, x0_mean, t_span, meas_error, sampling_rate=sampling_rate)

ts = jnp.linspace(*t_span, len(true_traj))

plt.figure()
plt.plot(observations[:, 0], observations[:, 1])
plt.xlabel(r"$q$")
plt.ylabel(r"$p$")
plt.title("Phase space trajectory")
plt.show()

plt.figure()
plt.plot(ts[1:], observations[:, 0], '.', label=r"$q$ (measured)")
plt.plot(ts[1:], observations[:, 1], '.', label=r"$p$ (measured)")
plt.plot(ts, true_traj[:, 0], label=r"$q$")
plt.plot(ts, true_traj[:, 1], label=r"$p$")
plt.title("Trajectory")
plt.xlabel("Time")
plt.legend()
plt.show()

energies = jax.vmap(hamiltonian, in_axes=(0, None))(observations, true_params)
plt.figure()
plt.plot(ts[1:], energies)
plt.title("Energy vs time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.show()


####################
# Filtering
####################
def drift_fun(x, params):
    """
    The drift function of the augmented state.
    """
    q, p = x
    g = 9.81
    m, l = params[:2]
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q)])


def get_Q(params, dt):
    """
    Get the process noise covariance matrix `Q`
    by first defining the diffusion vector `L`.
    """
    q = jnp.diag(params[2:])
    return q * dt


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([1.5, 0.])
    x0cov = jnp.eye(2)
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(params, observations, dt, meas_error, smooth=False):
    """
    Wrapper function to get the marginal data log-likelihood
    and the filtered states.
    """
    # Define the transition model.
    Q = get_Q(params, dt)
    transition_model = FunctionalModel(
        lambda x: x + drift_fun(x, params) * dt,
        MVNStandard(jnp.zeros(2), Q)
    )

    # Define the observation model.
    R = jnp.diag(meas_error)
    H = jnp.eye(2)
    observation_model = FunctionalModel(
        lambda x: H @ x,
        MVNStandard(jnp.zeros(2), R)
    )

    # Get the initial state distribution and run the filter.
    x0 = get_x0(params)
    filt_states, ell = filtering(observations, x0, transition_model, observation_model, extended)

    if smooth:
        smoothed_states = smoothing(transition_model, filt_states, extended)
        return ell, filt_states, smoothed_states

    return ell, filt_states


####################
# Parameter estimation
####################
get_neg_log_lik = lambda params: -get_ell_and_filter(params, observations, dt, meas_error)[0]
grad_log_lik = jax.jit(jax.value_and_grad(get_neg_log_lik))


# Using L-BFGS-B
def wrapper_func(params):
    loss, grad_val = grad_log_lik(params)
    return np.array(loss, dtype=np.float64), np.array(grad_val, dtype=np.float64)


guess_params = np.array([1.5, 1., 0.01, 0.1])
opt_result = minimize(wrapper_func, guess_params, method='L-BFGS-B', jac=True, bounds=[(1e-3, None), (1e-3, None), (1e-3, None), (1e-3, None)])
best_params = opt_result.x

log_lik, filt_states, smoothed_states = get_ell_and_filter(best_params, observations, dt, meas_error, True)
print(f"The best parameters are: {best_params} with log-likelihood {log_lik:.4f}.")

plt.figure()
plt.plot(ts, smoothed_states.mean[:, 0], label=r"$q$ smoothed")
plt.plot(ts, smoothed_states.mean[:, 1], label=r"$p$ smoothed")
plt.plot(ts[1:], observations[:, 0], '.', label=r"$q$ measured")
plt.plot(ts[1:], observations[:, 1], '.', label=r"$p$ measured")
plt.plot(ts, true_traj[:, 0], label=r"$q$")
plt.plot(ts, true_traj[:, 1], label=r"$p$")
plt.title("Trajectory after parameter optimization.")
plt.xlabel("Time")
plt.legend()
plt.show()


# Using Adam
# def estimate_params(params, display=False):
#     num_epochs = 200
#     learning_rate = 0.01
#     optimizer = optax.adam(learning_rate)
#     opt_state = optimizer.init(params)
#
#     @jax.jit
#     def train_step(params, optimizer_state):
#         log_lik, grads = grad_log_lik(params)
#         updates, opt_state = optimizer.update(grads, optimizer_state)
#         return optax.apply_updates(params, updates), opt_state, log_lik
#
#     # Training loop
#     for i in range(num_epochs):
#         params, opt_state, nll = train_step(params, opt_state)
#         if display:
#             print(f"Epoch {i:4d} | Negative log-likelihood: {nll:.4f}")
#
#     if display:
#         print(f"Final value of parameters: {guess_params}")
#         print(f"True value of parameters: {true_params}")
#
#     return params, nll
#
#
# param_list, nll_list = [], []
# for seed in range(5):
#     print(f"Iteration {seed + 1}...")
#     key = random.PRNGKey(seed)
#     subkeys = random.split(key, 5)
#     guess_params_dict = {
#         'mass': random.uniform(subkeys[1], minval=0., maxval=3.),
#         'length': random.uniform(subkeys[2], minval=0., maxval=3.),
#         'q1': random.uniform(subkeys[3], minval=0., maxval=.1),
#         'q2': random.uniform(subkeys[4], minval=0., maxval=1.)
#     }
#     guess_params = jnp.array([value for value in guess_params_dict.values()])
#     params, nll = estimate_params(guess_params, True)
#     param_list.append(params)
#     nll_list.append(nll)
#
# best_params = param_list[np.argmin(nll_list)]
