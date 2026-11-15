"""Cross-entropy style energies keep an exact gradient on saturated predictions.

jnp.clip passes no gradient outside its range, so clipping z_mu to [eps, 1]
meant a prediction below eps (a true-class probability under 1e-7, a logit gap
near 16) sent no learning signal. The energies now floor the log in the value
only: values are unchanged, and the gradient is that of the unfloored log down
to the smallest normal float.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.energy import (
    BernoulliEnergy,
    CrossEntropyEnergy,
    KLDivergenceEnergy,
)

EPS = 1e-7
CFG = {"eps": EPS}
F32 = jnp.float32
ONEHOT = jnp.array([[0.0, 1.0, 0.0]], dtype=F32)


def _logits(gap):
    """True class (index 1) has logit 0; a wrong class leads by ``gap``."""
    return jnp.array([[gap, 0.0, 0.0]], dtype=F32)


class TestCrossEntropySaturation:
    @staticmethod
    def _loss(logits):
        mu = jax.nn.softmax(logits)
        return CrossEntropyEnergy.energy(ONEHOT, mu, CFG)[0]

    @pytest.mark.parametrize("gap", [5.0, 16.0, 17.0, 20.0, 30.0, 60.0])
    def test_gradient_matches_log_softmax(self, gap):
        want = jax.grad(lambda lg: -jnp.sum(ONEHOT * jax.nn.log_softmax(lg)))(
            _logits(gap)
        )
        got = jax.grad(self._loss)(_logits(gap))
        assert jnp.allclose(got, want, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("gap", [17.0, 30.0])
    def test_value_stays_floored(self, gap):
        assert float(self._loss(_logits(gap))) == pytest.approx(
            -math.log(EPS), rel=1e-4
        )

    @pytest.mark.parametrize("p", [1e-6, 1e-3, 0.3, 1.0])
    def test_value_is_unchanged_above_eps(self, p):
        z = jnp.array([[1.0, 0.0, 0.0]], dtype=F32)
        mu = jnp.array([[p, (1 - p) / 2, (1 - p) / 2]], dtype=F32)
        got = CrossEntropyEnergy.energy(z, mu, CFG)[0]
        assert float(got) == pytest.approx(-math.log(p), rel=1e-5, abs=1e-6)

    def test_exact_zero_probability_is_finite(self):
        z = jnp.array([[1.0, 0.0, 0.0]], dtype=F32)
        mu = jnp.array([[0.0, 1.0, 0.0]], dtype=F32)
        value, grad = jax.value_and_grad(
            lambda m: CrossEntropyEnergy.energy(z, m, CFG)[0]
        )(mu)
        assert bool(jnp.isfinite(value)) and bool(jnp.all(jnp.isfinite(grad)))


class TestBernoulliSaturation:
    @pytest.mark.parametrize("logit", [-5.0, -17.0, -20.0, -30.0, -60.0])
    def test_gradient_when_target_is_one_and_prediction_tiny(self, logit):
        z = jnp.array([[1.0]], dtype=F32)

        def loss(lg):
            return BernoulliEnergy.energy(z, jax.nn.sigmoid(lg), CFG)[0]

        got = jax.grad(loss)(jnp.array([[logit]], dtype=F32))
        want = jax.nn.sigmoid(logit) - 1.0  # d/dl of -log sigmoid(lg)
        assert jnp.allclose(got, want, rtol=1e-3)


class TestKLSaturation:
    @pytest.mark.parametrize("gap", [5.0, 17.0, 30.0])
    def test_gradient_with_a_one_hot_target(self, gap):
        def loss(lg):
            return KLDivergenceEnergy.energy(ONEHOT, jax.nn.softmax(lg), CFG)[0]

        got = jax.grad(loss)(_logits(gap))
        want = jax.nn.softmax(_logits(gap)) - ONEHOT
        assert jnp.allclose(got, want, rtol=1e-3, atol=1e-3)
