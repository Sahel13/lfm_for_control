import jax.numpy as jnp
import chex

from parsmooth._base import MVNStandard, FunctionalModel
import stoch_ham.continuous_discrete_filtering as cont_disc


class TestContDisc:
    f = lambda x: jnp.array([x[1]**2, -x[0]])
    q = MVNStandard(jnp.array([0., 0.]), 0.1 * jnp.eye(2))
    model = FunctionalModel(f, q)
    x = MVNStandard(jnp.array([0., 1.]), jnp.eye(2))
    dt = 0.1

    def test_mean_dynamics(self):
        approx = cont_disc._mean_dynamics(self.model, self.x)
        chex.assert_equal_shape((approx, self.x.mean))

    def test_cov_dynamics(self):
        dPdt, dCdt = cont_disc._cov_dynamics(self.model, self.x)
        chex.assert_equal_shape((dPdt, dCdt, self.x.cov))

    def test_joint_dynamics(self):
        pred_x, dCdt = cont_disc._joint_dynamics(self.model, self.x)
        chex.assert_trees_all_equal_shapes(self.x, pred_x)

    def test_rk4_step(self):
        f = lambda xi: cont_disc._joint_dynamics(self.model, xi)[0]
        pred_x = cont_disc.rk4_step(f, self.x, self.dt)
        chex.assert_trees_all_equal_shapes(self.x, pred_x)
