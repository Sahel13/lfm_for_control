import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from stoch_ham.base import MVNStandard, FunctionalModel
from stoch_ham.filtering import filtering
from stoch_ham.linearization import extended
from stoch_ham.simple_pendulum.data import get_dataset, hamiltonian

import scipy
from scipy.optimize import minimize

####################
# Get the data
####################
seed = 10
key = random.PRNGKey(seed)

meas_error = jnp.array([0.1, 0.5])

true_params = {
    'mass': 1.,
    'length': 2.,
    'lambda': 5.,
    'q': 0.5,
    'meas_error': meas_error
}

sim_dt = 0.001
sampling_rate = 100
dt = 1./sampling_rate

x0_mean = jnp.array([jnp.pi / 2, 0.])
t_span = (0., 10.)

true_traj, observations = get_dataset(
    key, 1, true_params, x0_mean, t_span, sim_dt, sampling_rate)[0]

ts = jnp.linspace(*t_span, len(observations))

plot_figures = False

if plot_figures:
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
# The parameters are mass, length, b, lambda, q1, q2.
guess_params = jnp.array([1., 2., 0., 5., .5, .01])


def drift_fun(x, params):
    """
    The drift function of the augmented state.
    """
    q, p, u = x
    g = 9.81
    m, l, b, lamba = params[:4]
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) - b * p + u, -lamba * u])


def get_transition_noise(params, dt):
    """
    Get the process noise covariance matrix `Q`
    by first defining the diffusion vector `L`.
    """
    L = jnp.array([[0., 0.], [1., 0.], [0., 1.]])
    diffusion_matrix = jnp.diag(params[4:])
    return L @ diffusion_matrix @ L.T * dt


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([jnp.pi / 2, 0., 0.])
    u0_cov = params[-1] / (2 * params[3])
    x0cov = jnp.diag(jnp.array([1., 1., u0_cov]))
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(params, dt, meas_error):
    """
    Wrapper function to get the marginal data log-likelihood
    and the filtered states.
    """
    # Define the transition model.
    Q = get_transition_noise(params, dt)
    transition_model = FunctionalModel(
        lambda x: x + drift_fun(x, params) * dt,
        MVNStandard(jnp.zeros(3), Q)
    )

    # Define the observation model.
    R = meas_error
    H = jnp.array([[1., 0., 0.], [0., 1., 0.]])
    observation_model = FunctionalModel(
        lambda x: H @ x,
        MVNStandard(jnp.zeros(2), R)
    )

    # Get the initial state distribution and run the filter.
    x0 = get_x0(params)
    filt_states, ell = filtering(observations, x0, transition_model, observation_model, extended)
    return ell, filt_states


ell, filt_states = get_ell_and_filter(guess_params, dt, meas_error)
print(ell)
print(filt_states.mean.shape)

####################
# Parameter estimation
####################
get_neg_log_lik = lambda params: -get_ell_and_filter(params, dt)[0]
grad_log_lik = jax.value_and_grad(get_neg_log_lik)

plt.figure()
plt.plot(ts, filt_states.mean[1:, 0], label=r"$q$ filtered")
plt.plot(ts, filt_states.mean[1:, 1], label=r"$p$ filtered")
plt.plot(ts, true_traj[:, 0], label=r"$q$")
plt.plot(ts, true_traj[:, 1], label=r"$p$")
plt.title("Trajectory")
plt.xlabel("Time")
plt.legend()
plt.show()


def rmse(x, y):
    """
    Returns root mean square error between two vectors x and y.
    """
    return jnp.sqrt(jnp.mean(jnp.square(x - y),  axis=0))


print(filt_states.mean[:10, 0])
print(true_traj[:20, 0])
