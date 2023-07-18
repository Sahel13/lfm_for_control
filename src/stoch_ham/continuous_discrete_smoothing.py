import jax

from parsmooth._base import MVNStandard
from parsmooth._utils import none_or_shift, none_or_concat


def smoothing(filtered_trajectory, predicted_trajectory, gains):
    last_state = jax.tree_map(lambda z: z[-1], filtered_trajectory)

    def body(smoothed, inp):
        filtered, predicted, gain = inp
        mean = filtered.mean + gain @ (smoothed.mean - predicted.mean)
        cov = filtered.cov + gain @ (smoothed.cov - predicted.cov) @ gain.T
        smoothed_state = MVNStandard(mean, cov)
        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(body,
                                      last_state,
                                      [none_or_shift(filtered_trajectory, -1), predicted_trajectory, gains],
                                      reverse=True)

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states
