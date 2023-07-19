import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from data import add_meas_noise
from parsmooth._base import MVNStandard, FunctionalModel
from stoch_ham.continuous_discrete_filtering import filtering
from stoch_ham.continuous_discrete_smoothing import smoothing

seed = 2
key = random.PRNGKey(seed)

########################################
# Get the data
########################################
data = jnp.array(np.loadtxt("trajectory_data.csv", dtype="float64", delimiter=","))
init_state = jnp.array([1., -0.01, 0.])
true_traj = jnp.vstack([init_state[None, :], data])

meas_error = jnp.array([.2, .4])
key, subkey = random.split(key)
observations = add_meas_noise(subkey, true_traj[:, :2], meas_error)

t_span = (0., 5.)
dt = 0.05
t_eval = jnp.arange(t_span[0], t_span[1] + dt, dt)

fig, axs = plt.subplots(3, 1, sharex=True, layout="tight")
axs[0].plot(t_eval[1:], observations[:, 0], '.', label="Measured")
axs[0].plot(t_eval, true_traj[:, 0], label="True")
axs[0].set_ylabel("Position")
axs[1].plot(t_eval[1:], observations[:, 1], '.', label="Measured")
axs[1].plot(t_eval, true_traj[:, 1], label="True")
axs[1].set_ylabel("Velocity")
axs[2].plot(t_eval, true_traj[:, 2], label="True")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Control input")
axs[2].legend()
plt.show()


########################################
# Filtering and smoothing
########################################
def drift_fun(x, params):
    """
    The drift function of the augmented state.
    """
    # p is the angular velocity.
    q, p, u = x
    m, l, lamba = params[:3]
    g = 9.81
    damping = 1e-3
    return jnp.array([p, -g / l * jnp.sin(q) + (u - damping * p) / (m * l**2), -lamba * u])


def get_LQL(params):
    """
    LQL term of the covariance.
    """
    eps = 1e-4  # If needed to prevent noise covariance from becoming singular.
    LQL = jnp.diag(jnp.array([eps, eps, params[-1]]))
    return LQL


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([1., -0.01, 0.])
    u0_cov = params[-1] / (2 * params[2])
    x0cov = jnp.diag(jnp.array([1., 1., u0_cov]))
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(raw_params, observations, dt, meas_error, smooth=False):
    """
    Wrapper function to get the marginal data log-likelihood
    and the filtered/smoothed states.
    """
    params = jax.nn.softplus(raw_params)
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
    filtered_traj, ell, predicted_traj, gains = filtering(
        observations, x0, transition_model, observation_model, dt
    )

    if smooth:
        smoothed_traj = smoothing(filtered_traj, predicted_traj, gains)
        return ell, filtered_traj, smoothed_traj

    return ell, filtered_traj


########################################
# Parameter estimation
########################################
# Define helper methods for optimization.
get_neg_log_lik = lambda params: -get_ell_and_filter(params, observations, dt, meas_error)[0]
grad_log_lik = jax.value_and_grad(get_neg_log_lik)


# Using Adam
def estimate_params(params, display=False):
    num_epochs = 100
    learning_rate = .1
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state):
        nll, grads = grad_log_lik(params)
        updates, opt_state = optimizer.update(grads, optimizer_state)
        return optax.apply_updates(params, updates), opt_state, nll

    # Training loop
    for i in range(num_epochs):
        params, opt_state, nll = train_step(params, opt_state)
        if display:
            print(f"Epoch {i:4d} | Negative log-likelihood: {nll:.4f}")

    if display:
        print(f"Final value of parameters: {jax.nn.softplus(params)}")

    return params, nll


num_trials = 3
key, subkey = random.split(key)
init_values = random.uniform(subkey, (num_trials, 4), minval=0., maxval=3.)

# For testing
# num_trials = 1
# init_values = jnp.array([1., 2., 1., 1.])[None, :]

param_list, nll_list = [], []
for i in range(num_trials):
    print(f"Trial {i + 1}...")
    guess_params = init_values[i]
    params, nll = estimate_params(guess_params, display=True)

    # Skip nans
    if jnp.isnan(nll):
        continue

    param_list.append(params)
    nll_list.append(nll)

# Obtain the smoothed trajectory using the best parameters.
best_params = param_list[np.argmin(nll_list)]
log_lik, filtered_traj, smoothed_traj = get_ell_and_filter(best_params, observations, dt, meas_error, True)
print(f"The best parameters are: {jax.nn.softplus(best_params)} with log-likelihood {log_lik:.4f}.")


########################################
# Visualize the results.
########################################
def get_std(x: MVNStandard):
    """Compute the standard deviations of the smoothed trajectory."""
    stds = jnp.sqrt(jnp.diagonal(x.cov, axis1=1, axis2=2)).T
    return jnp.stack((x.mean.T - 2 * stds, x.mean.T + 2 * stds), axis=1)


smoothed_covs = get_std(smoothed_traj)

fig, axs = plt.subplots(3, 1, sharex=True, layout="tight")
axs[0].plot(t_eval, true_traj[:, 0], label="True")
axs[0].scatter(t_eval[1:], observations[:, 0], label="Measured")
axs[0].plot(t_eval, smoothed_traj.mean[:, 0], label="Smoothed")
axs[0].fill_between(t_eval, *smoothed_covs[0], alpha=0.2, label="2 std")
axs[0].set_ylabel(r"$q$")
axs[0].legend()

axs[1].plot(t_eval, true_traj[:, 1], label="True")
axs[1].scatter(t_eval[1:], observations[:, 1], label="Measured")
axs[1].plot(t_eval, smoothed_traj.mean[:, 1], label="Smoothed")
axs[1].fill_between(t_eval, *smoothed_covs[1], alpha=0.2, label="2 std")
axs[1].set_ylabel(r"$p$")
axs[1].legend()

axs[2].plot(t_eval, true_traj[:, 2], label="True")
axs[2].plot(t_eval, smoothed_traj.mean[:, 2], label="Smoothed", color="orange")
axs[2].fill_between(t_eval, *smoothed_covs[2], alpha=0.2, label="2 std", color="orange")
axs[2].set_ylabel(r"$u$")
axs[2].set_xlabel("Time t")
axs[2].legend()

fig.suptitle("Smoothed trajectory")
plt.show()
