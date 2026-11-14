"""Convexity bound for StorkeyHopfield's latent energy.

For a Gaussian PC term with precision p, the Hessian of E_pc + s * E_hop with
respect to z is p I + (s / D) (W W^T - (W + W^T) / 2), which is bounded below
by (p - s / (4 D)) I for every W. ``convex_strength_limit`` is the strength
4 D p at which that bound reaches zero.
"""

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.types import NodeState
from fabricpc.nodes.storkey_hopfield import StorkeyHopfield, convex_strength_limit

D = 8


def _energy(z, W, strength, precision=1.0):
    """The node's own energy() override at latent z, with z_mu = 0."""
    info = SimpleNamespace(energy=GaussianEnergy(precision=precision))
    z_mu = jnp.zeros((1, z.shape[-1]))
    state = NodeState(
        z_latent=z[None],
        z_mu=z_mu,
        error=z[None] - z_mu,
        energy=jnp.zeros((1,)),
        latent_grad=jnp.zeros_like(z_mu),
    )
    return StorkeyHopfield.energy(None, {}, state, (W, strength), info)[0]


def _lambda_min(W, strength, precision=1.0, seed=0):
    z = jax.random.normal(jax.random.PRNGKey(seed), (W.shape[0],))
    hessian = jax.hessian(lambda v: _energy(v, W, strength, precision))(z)
    return float(np.linalg.eigvalsh(np.asarray(0.5 * (hessian + hessian.T))).min())


class TestConvexStrengthLimit:
    def test_value(self):
        assert convex_strength_limit(8) == 32.0
        assert convex_strength_limit(8, precision=2.5) == 80.0

    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("scale", [0.05, 0.3, 1.0])
    @pytest.mark.parametrize("strength", [0.5, 5.0, 20.0, 60.0])
    def test_hessian_lower_bound_holds_for_any_W(self, symmetric, scale, strength):
        """lambda_min(H_z) >= p - s / (4 D), symmetric or not."""
        rng = np.random.RandomState(0)
        W = jnp.asarray(rng.randn(D, D) * scale, dtype=jnp.float32)
        if symmetric:
            W = 0.5 * (W + W.T)
        precision = 1.0
        bound = precision - strength / (4 * D)
        assert _lambda_min(W, strength, precision) >= bound - 1e-4

    def test_bound_is_attained_at_half_identity(self):
        """W = I / 2 reaches the bound, so no larger limit is valid."""
        W = 0.5 * jnp.eye(D)
        limit = convex_strength_limit(D)
        below, above = 0.9 * limit, 1.1 * limit
        assert _lambda_min(W, below) == pytest.approx(1.0 - below / (4 * D), abs=1e-4)
        assert _lambda_min(W, above) == pytest.approx(1.0 - above / (4 * D), abs=1e-4)
        assert _lambda_min(W, above) < 0.0

    def test_limit_scales_with_precision(self):
        W = 0.5 * jnp.eye(D)
        precision = 2.5
        strength = 0.9 * convex_strength_limit(D, precision)
        assert _lambda_min(W, strength, precision) > 0.0
        assert _lambda_min(W, 1.1 * convex_strength_limit(D, precision), precision) < 0

    def test_strength_gradient_is_negative_at_an_attractor(self):
        """At W = I / 2, E_hop = -|z|^2 / (8 D) < 0, so dE/ds < 0.

        A learnable strength therefore receives a gradient that increases it
        for as long as W keeps z at an attractor.
        """
        W = 0.5 * jnp.eye(D)
        z = jax.random.normal(jax.random.PRNGKey(3), (D,))
        grad = float(jax.grad(lambda s: _energy(z, W, s))(1.0))
        assert grad == pytest.approx(-float(z @ z) / (8 * D), rel=1e-5)
        assert grad < 0.0


class TestFixedStrengthGuard:
    def _build(self, **kwargs):
        return StorkeyHopfield(shape=(D,), name="hop", **kwargs)

    def test_warns_at_the_limit(self):
        with pytest.warns(UserWarning, match="non-convex"):
            self._build(hopfield_strength=convex_strength_limit(D))

    def test_warns_above_the_limit(self):
        with pytest.warns(UserWarning, match="non-convex"):
            self._build(hopfield_strength=2 * convex_strength_limit(D))

    def test_silent_below_the_limit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build(hopfield_strength=0.99 * convex_strength_limit(D))

    def test_silent_for_learnable_strength(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build()

    def test_limit_uses_the_energy_precision(self):
        precision = 3.0
        energy = GaussianEnergy(precision=precision)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build(
                energy=energy,
                hopfield_strength=0.99 * convex_strength_limit(D, precision),
            )
        with pytest.warns(UserWarning, match="non-convex"):
            self._build(
                energy=energy,
                hopfield_strength=convex_strength_limit(D, precision),
            )

    def test_silent_for_non_gaussian_energy(self):
        """The bound is derived for a Gaussian PC term only."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build(
                energy=CrossEntropyEnergy(),
                hopfield_strength=10 * convex_strength_limit(D),
            )
