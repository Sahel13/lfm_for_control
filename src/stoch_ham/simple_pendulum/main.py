import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from stoch_ham.base import MVNStandard, FunctionalModel
from stoch_ham.filtering import filtering
from stoch_ham.linearization import extended
from stoch_ham.simple_pendulum.data import get_dataset, hamiltonian

import optax

####################
# Get the data
####################
seed = 5
key = random.PRNGKey(seed)

meas_error = jnp.array([.5, 2.5])

true_params = {
    'mass': 1.,
    'length': 2.,
    'lambda': 5.,
    'q': 0.5
}

sim_dt = 0.001
sampling_rate = 10
dt = 1./sampling_rate

x0_mean = jnp.array([jnp.pi / 2, 0.])
t_span = (0., 10.)

true_traj, observations = get_dataset(
    key, 1, true_params, x0_mean, t_span, meas_error, sim_dt, sampling_rate)[0]

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
def drift_fun(x, params):
    """
    The drift function of the augmented state.
    """
    q, p, u = x
    g = 9.81
    m, l, lamba = params['mass'], params['length'], params['lambda']
    return jnp.array([p / (m * l ** 2), -m * g * l * jnp.sin(q) - u, -lamba * u])


def get_Q(params, dt):
    """
    Get the process noise covariance matrix `Q`
    by first defining the diffusion vector `L`.
    """
    L = jnp.array([0., 0., 1.])[:, None]
    return L @ L.T * params['q'] * dt


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([jnp.pi / 2, 0., 0.])
    u0_cov = params['q'] / (2 * params['lambda'])
    x0cov = jnp.diag(jnp.array([1., 1., u0_cov]))
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(params, observations, dt, meas_error):
    """
    Wrapper function to get the marginal data log-likelihood
    and the filtered states.
    """
    # Define the transition model.
    Q = get_Q(params, dt)
    transition_model = FunctionalModel(
        lambda x: x + drift_fun(x, params) * dt,
        MVNStandard(jnp.zeros(3), Q)
    )

    # Define the observation model.
    R = jnp.diag(meas_error)
    H = jnp.array([[1., 0., 0.], [0., 1., 0.]])
    observation_model = FunctionalModel(
        lambda x: H @ x,
        MVNStandard(jnp.zeros(2), R)
    )

    # Get the initial state distribution and run the filter.
    x0 = get_x0(params)
    filt_states, ell = filtering(observations, x0, transition_model, observation_model, extended)
    return ell, filt_states


####################
# Parameter estimation
####################
get_neg_log_lik = lambda params: -get_ell_and_filter(params, observations, dt, meas_error)[0]
grad_log_lik = jax.value_and_grad(get_neg_log_lik)


def estimate_params(params, display=False):
    num_epochs = 2000
    schedule = optax.piecewise_constant_schedule(0.01, {600: 0.1})
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state):
        log_lik, grads = grad_log_lik(params)
        updates, opt_state = optimizer.update(grads, optimizer_state)
        return optax.apply_updates(params, updates), opt_state, log_lik

    # Training loop
    for i in range(num_epochs):
        params, opt_state, nll = train_step(params, opt_state)
        if display:
            print(f"Epoch {i:4d} | Negative log-likelihood: {nll:.4f}")

    if display:
        print(f"Final value of parameters: {guess_params}")
        print(f"True value of parameters: {true_params}")

    return params, nll


param_list, nll_list = [], []
for seed in range(4, 5):
    print(f"Iteration {seed + 1}...")
    key = random.PRNGKey(seed)
    subkeys = random.split(key, 5)
    guess_params = {
        'mass': random.uniform(subkeys[1], minval=0., maxval=2.),
        'length': random.uniform(subkeys[2], minval=0., maxval=5.),
        'lambda': random.uniform(subkeys[3], minval=3., maxval=7.),
        'q': random.uniform(subkeys[4], minval=0., maxval=1.)
    }
    params, nll = estimate_params(guess_params)
    param_list.append(params)
    nll_list.append(nll)

nll, filt_states = get_ell_and_filter(param_list[0], observations, dt, meas_error)
print(nll)

plt.figure()
plt.plot(ts, filt_states.mean[1:, 0], label=r"$q$ filtered")
plt.plot(ts, filt_states.mean[1:, 1], label=r"$p$ filtered")
plt.plot(ts, observations[:, 0], '.', label=r"$q$ (measured)")
plt.plot(ts, observations[:, 1], '.', label=r"$p$ (measured)")
plt.plot(ts, true_traj[:, 0], label=r"$q$")
plt.plot(ts, true_traj[:, 1], label=r"$p$")
plt.title("Trajectory after parameter optimization.")
plt.xlabel("Time")
plt.legend()
plt.show()
