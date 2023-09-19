import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from parsmooth._base import MVNStandard, FunctionalModel
from stoch_ham.filtering_for_control import filtering
from parsmooth.linearization import extended
from parsmooth.methods import smoothing


def joint_dynamics(state, params, dt):
    x, u = state[:-1], state[-1]
    Lambda, q = params
    # For x
    F_x = jnp.array([[1., dt], [0., 1.]])
    F_u = jnp.array([0., dt])
    new_x = F_x @ x + F_u * u
    # For u
    a = jnp.exp(-Lambda * dt)
    new_u = a * u
    return jnp.concatenate([new_x, new_u.reshape(1)])


def joint_covs(params, dt):
    V = jnp.array([[1e-4, 0.], [0., 1e-4]])
    Lambda, q = params
    sigma = q / (2 * Lambda) * (1 - jnp.exp(-2 * Lambda * dt))
    return jax.scipy.linalg.block_diag(V, sigma)


def observation_model():
    eta = 1.
    cov_inv = 2 * eta * jnp.diag(jnp.array([10., 1., 1.]))
    return MVNStandard(jnp.zeros(3), jnp.linalg.inv(cov_inv))


def get_x0(params):
    """
    Define the distribution of the initial state.
    """
    x0mean = jnp.array([1.0, 2.0, 0.])
    u0_cov = params[-1] / (2 * params[2])
    x0cov = jnp.diag(jnp.array([1e-4, 1e-4, u0_cov]))
    x0 = MVNStandard(x0mean, x0cov)
    return x0


def get_ell_and_filter(params, t_span, dt, smooth=False):
    """
    Wrapper function to get the marginal data log-likelihood
    and the smoothed states.
    """
    # Define the transition model.
    W = joint_covs(params, dt)
    transition_model = FunctionalModel(
        lambda x: joint_dynamics(x, params, dt),
        MVNStandard(jnp.zeros(3), W)
    )

    # Get the initial state distribution and run the filter.
    x0 = get_x0(params)
    num_steps = int((t_span[1] - t_span[0]) / dt)
    observations = jnp.zeros((num_steps, 3))
    trajectory = filtering(observations, x0, transition_model, observation_model(), None)

    if smooth:
        return trajectory, smoothing(transition_model, trajectory, extended, parallel=False)

    return trajectory


dt = 0.1
t_span = (0., 5.)
params = jnp.array([1., 1.])

filtered_traj, smoothed_traj = get_ell_and_filter(params, t_span, dt, smooth=True)

# Plot the results.
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(filtered_traj.mean[:, 0], label='Filtered')
axs[0].plot(smoothed_traj.mean[:, 0], label='Smoothed')
axs[0].legend()
axs[0].set_ylabel('Position')
axs[1].plot(filtered_traj.mean[:, 1], label='Filtered')
axs[1].plot(smoothed_traj.mean[:, 1], label='Smoothed')
axs[1].legend()
axs[1].set_ylabel('Velocity')
axs[2].plot(filtered_traj.mean[:, 2], label='Filtered')
axs[2].plot(smoothed_traj.mean[:, 2], label='Smoothed')
axs[2].legend()
axs[2].set_ylabel('Control')
axs[2].set_xlabel('Time')
plt.show()
