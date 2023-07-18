import jax
import jax.numpy as jnp
import jax.random as random
import optax
import numpy as np

from data import add_meas_noise
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from parsmooth._base import MVNStandard, FunctionalModel
from stoch_ham.continuous_discrete_filtering import filtering
from stoch_ham.continuous_discrete_smoothing import smoothing

########################################
# Get the data
########################################
seed = 0
key = random.PRNGKey(seed)

meas_error = jnp.array([.5, 2.5, 0.])

true_params = {
    'mass': 1.,
    'length': 2.,
    'a': 3.,
    'omega': jnp.pi / 2
}

sampling_rate = 20
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
# Filtering and smoothing
########################################
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
    LQL term of the covariance.
    """
    eps = 0.  # To prevent noise covariance from becoming singular.
    LQL = jnp.diag(jnp.array([eps, eps, params[-1]]))
    return LQL


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([1.5, 0., 0.])
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
    num_epochs = 200
    learning_rate = 0.01
    schedule = optax.piecewise_constant_schedule(learning_rate, {100: 0.5, 150: 0.1})
    optimizer = optax.adam(learning_rate=schedule)
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


# num_trials = 3
# key, subkey = random.split(key)
# init_values = random.uniform(subkey, (num_trials, 4), minval=0., maxval=2.)

# For testing
num_trials = 1
init_values = jnp.array([1., 2., 1., 1.])[None, :]

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
# Parameter estimation
########################################
def get_std(x: MVNStandard):
    """Compute the standard deviations of the smoothed trajectory."""
    stds = jnp.sqrt(jnp.diagonal(x.cov, axis1=1, axis2=2)).T
    return jnp.stack((x.mean.T - 2 * stds, x.mean.T + 2 * stds), axis=1)


smoothed_covs = get_std(smoothed_traj)

fig, axs = plt.subplots(3, 1, sharex=True, layout="tight")
axs[0].plot(ts, true_traj[:, 0], label="True")
axs[0].scatter(ts[1:], observations[:, 0], label="Measured")
axs[0].plot(ts, smoothed_traj.mean[:, 0], label="Smoothed")
axs[0].fill_between(ts, *smoothed_covs[0], alpha=0.2, label="2 std")
axs[0].set_ylabel(r"$q$")
axs[0].legend()

axs[1].plot(ts, true_traj[:, 1], label="True")
axs[1].scatter(ts[1:], observations[:, 1], label="Measured")
axs[1].plot(ts, smoothed_traj.mean[:, 1], label="Smoothed")
axs[1].fill_between(ts, *smoothed_covs[1], alpha=0.2, label="2 std")
axs[1].set_ylabel(r"$p$")
axs[1].legend()

axs[2].plot(ts, true_traj[:, 2], label="True")
axs[2].plot(ts, smoothed_traj.mean[:, 2], label="Smoothed", color="orange")
axs[2].fill_between(ts, *smoothed_covs[2], alpha=0.2, label="2 std", color="orange")
axs[2].set_ylabel(r"$u$")
axs[2].set_xlabel("Time t")
axs[2].legend()

fig.suptitle("Smoothed trajectory")
plt.show()
