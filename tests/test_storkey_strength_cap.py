"""Opt-in cap on the learnable StorkeyHopfield strength.

With max_strength = c the effective strength is c * sigmoid(raw), which is
below c for every raw. When c < 4 D p the latent energy is convex for every W
(see convex_strength_limit), so the learnable strength cannot run past the
convexity limit.
"""

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.initializers import XavierInitializer, ZerosInitializer
from fabricpc.core.types import NodeParams, NodeState
from fabricpc.nodes.storkey_hopfield import StorkeyHopfield, convex_strength_limit

D = 8


def _config(cap=None, strength=None):
    return {
        "hopfield_strength": strength,
        "max_strength": cap,
        "use_bias": False,
        "enforce_symmetry": True,
        "zero_diagonal": False,
        "bias_init": ZerosInitializer(),
    }


def _params(cap=None):
    return StorkeyHopfield.initialize_params(
        jax.random.PRNGKey(0), (D,), {"in": (D,)}, XavierInitializer(), _config(cap)
    )


def _with_raw(params, raw):
    biases = {"hopfield_strength": jnp.asarray(raw, dtype=jnp.float32)}
    return NodeParams(weights=params.weights, biases=biases)


def _lambda_min(W, strength, precision=1.0):
    def energy(z):
        info = SimpleNamespace(energy=GaussianEnergy(precision=precision))
        z_mu = jnp.zeros((1, D))
        state = NodeState(
            z_latent=z[None],
            z_mu=z_mu,
            error=z[None] - z_mu,
            energy=jnp.zeros((1,)),
            latent_grad=jnp.zeros_like(z_mu),
        )
        return StorkeyHopfield.energy(None, {}, state, (W, strength), info)[0]

    z = jax.random.normal(jax.random.PRNGKey(0), (D,))
    hessian = jax.hessian(energy)(z)
    return float(np.linalg.eigvalsh(np.asarray(0.5 * (hessian + hessian.T))).min())


class TestEffectiveStrength:
    def test_default_is_softplus_and_starts_at_one(self):
        params = _params()
        assert float(StorkeyHopfield.effective_strength(params, _config())) == (
            pytest.approx(1.0, abs=1e-5)
        )
        got = StorkeyHopfield.effective_strength(_with_raw(params, 3.0), _config())
        assert float(got) == pytest.approx(float(jax.nn.softplus(3.0)), rel=1e-6)

    @pytest.mark.parametrize("cap", [2.0, 20.0, 100.0])
    def test_capped_starts_at_one(self, cap):
        got = StorkeyHopfield.effective_strength(_params(cap), _config(cap))
        assert float(got) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("raw", [-50.0, 0.0, 10.0, 1000.0])
    def test_capped_stays_in_zero_cap(self, raw):
        cap = 20.0
        params = _with_raw(_params(cap), raw)
        got = float(StorkeyHopfield.effective_strength(params, _config(cap)))
        assert 0.0 <= got <= cap

    def test_saturates_at_the_cap(self):
        cap = 20.0
        params = _with_raw(_params(cap), 1000.0)
        got = float(StorkeyHopfield.effective_strength(params, _config(cap)))
        assert got == pytest.approx(cap)

    def test_gradient_is_alive_at_init(self):
        cap = 20.0
        params = _params(cap)

        def s(raw):
            p = _with_raw(params, 0.0)
            p = NodeParams(weights=p.weights, biases={"hopfield_strength": raw})
            return StorkeyHopfield.effective_strength(p, _config(cap))

        raw0 = params.biases["hopfield_strength"]
        assert float(jax.grad(s)(raw0)) > 0.0

    def test_fixed_strength_is_returned_unchanged(self):
        params = NodeParams(weights=_params().weights, biases={})
        got = StorkeyHopfield.effective_strength(params, _config(strength=2.0))
        assert got == 2.0


class TestConvexAtTheCap:
    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("scale", [0.05, 0.3, 1.0])
    def test_hessian_is_positive_definite_at_saturation(self, symmetric, scale):
        cap = 0.9 * convex_strength_limit(D)
        rng = np.random.RandomState(1)
        W = jnp.asarray(rng.randn(D, D) * scale, dtype=jnp.float32)
        if symmetric:
            W = 0.5 * (W + W.T)
        params = _with_raw(_params(cap), 1000.0)
        s = StorkeyHopfield.effective_strength(params, _config(cap))
        assert _lambda_min(W, s) >= 1.0 - cap / (4 * D) - 1e-4 > 0.0

    def test_half_identity_is_convex_at_the_cap(self):
        cap = 0.9 * convex_strength_limit(D)
        got = _lambda_min(0.5 * jnp.eye(D), cap)
        assert got == pytest.approx(1.0 - cap / (4 * D), abs=1e-4)
        assert got > 0.0


class TestConstructorChecks:
    def _build(self, **kwargs):
        return StorkeyHopfield(shape=(D,), name="hop", **kwargs)

    def test_accepts_a_cap_below_the_limit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build(max_strength=0.99 * convex_strength_limit(D))

    def test_rejects_a_cap_at_or_above_the_limit(self):
        limit = convex_strength_limit(D)
        for cap in (limit, 2 * limit):
            with pytest.raises(ValueError, match="below"):
                self._build(max_strength=cap)

    def test_rejects_a_cap_not_above_the_initial_strength(self):
        for cap in (1.0, 0.5):
            with pytest.raises(ValueError, match="exceed"):
                self._build(max_strength=cap)

    def test_rejects_a_cap_with_a_fixed_strength(self):
        with pytest.raises(ValueError, match="fixed"):
            self._build(max_strength=10.0, hopfield_strength=2.0)

    def test_limit_uses_the_energy_precision(self):
        low = GaussianEnergy(precision=0.5)  # limit 16
        high = GaussianEnergy(precision=2.0)  # limit 64
        with pytest.raises(ValueError, match="below"):
            self._build(energy=low, max_strength=20.0)
        self._build(energy=high, max_strength=20.0)

    def test_non_gaussian_energy_skips_the_convexity_check(self):
        self._build(
            energy=CrossEntropyEnergy(), max_strength=10 * convex_strength_limit(D)
        )
