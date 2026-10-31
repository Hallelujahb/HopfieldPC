# StorkeyHopfield: per-sample energy reduction and a consistent E_hop formula

## Problem

`NodeBase.energy` is documented to return per-sample energy of shape `(batch,)`
(`fabricpc/nodes/base.py`, `energy()` docstring: "Returns: Per-sample energy,
shape (batch,)"), and `forward_and_latent_grads` sums it over the batch to a
scalar. The base energy functionals honor this by reducing over every non-batch
axis (`GaussianEnergy.energy`: `axes_to_sum = tuple(range(1, len(diff.shape)))`).

`StorkeyHopfield.energy` (`fabricpc/nodes/storkey_hopfield.py`, the
`E_hopfield = ... jnp.sum(wz * (wz - z), axis=-1)` line) reduces over the last
axis only. For a latent of shape `(batch, ..., D)` with extra axes the Hopfield
term therefore has shape `(batch, ...)` and is added to a `(batch,)` PC term.
This is a contract violation; it surfaces as a broadcast error, or as a
mis-shaped carry where the extra axis happens to equal the batch size. For
`(batch, D)` latents (the only shape exercised by `tests/test_storkey_hopfield.py`)
nothing changes.

Second, the documentation says `E_hop = (1/2D) z^T (W^2 - W) z`
(module docstring, `energy()` docstring, `docs/user_guides/10_api_nodes.md`), and
`test_attractor_energy_formula` pins that form. The code computes
`wz = z @ W; sum(wz * (wz - z))`, i.e. `(1/2D)(||W^T z||^2 - z^T W z)`. These are
the same for symmetric `W` (the default, `enforce_symmetry=True`) and differ for
non-symmetric `W`. The existing test only runs the default, so it cannot see the
difference. In the non-default mode the autodiff latent gradient is
`(s/D)(W W^T - (W + W^T)/2) z`, not the documented `(s/D)(W^2 - W) z`.

Intended outcome: `energy()` returns `(batch,)` for any latent rank, and the
docs, the test and the code describe the same function.

## Chosen approach

1. Reduce `E_hop` over `tuple(range(1, z.ndim))` (same convention as the base
   functionals). `D` stays `z.shape[-1]`, so the `1/D` scale is unchanged.
2. Keep the code's `||W^T z||^2 - z^T W z` form and correct the docs and the
   test to it, noting that it equals `W^2 - W` only for symmetric `W`.
3. New tests (`tests/test_storkey_hopfield.py::TestEnergyShapeAndGradient`):
   shape `(batch,)` for `(B,D)`, `(B,T,D)` with `B != T`, `B == T`, `B == 1`;
   sum-over-slices consistency; and a float64 `jax.grad` check against the
   analytic gradient for symmetric and non-symmetric `W` (scoped x64).

## Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **A (chosen)**: keep `||W^T z||^2` form, fix docs/test | No numerical change for the default; the form is a sound quadratic (`||.||^2 >= 0`); smallest diff | The energy is no longer literally the Storkey `W^2 - W` expression for non-symmetric `W` |
| B: change the code to `z^T (W^2 - W) z` | Matches the documented/plan formula for all `W` | Changes behaviour and gradients in the non-default mode; for non-symmetric `W` the quadratic form only sees the symmetric part of `W^2 - W`, so the gradient would need a symmetrisation decision; needs maintainer sign-off |
| C: enforce `enforce_symmetry=True` always | Removes the discrepancy | Removes a public option; breaking change |
| D: reduce with `axis=-1` and require callers to flatten | No code change | Pushes the contract onto every caller; base energies already reduce all axes |

Choice between A and B is a maintainer decision (see the issue); this PR
implements A.

## Migration inventory

Call sites of `StorkeyHopfield.energy` all go through `NodeBase.forward` /
`forward_with_aux`; no caller reads the Hopfield term separately.
`(batch, D)` latents produce identical values before and after (the reduction
axes coincide), so no existing demo or baseline changes.

| Item | Change |
|---|---|
| `fabricpc/nodes/storkey_hopfield.py` (module docstring, `energy()` docstring, reduction line) | updated |
| `docs/user_guides/10_api_nodes.md` energy block | updated |
| `tests/test_storkey_hopfield.py::test_attractor_energy_formula` | pinned to `W W^T` form (same value for the default symmetric `W`) |
| `docs/dev_plans_archive/storkey_hopfield_plan.md` and other archived plans | left as historical records |
| `scripts/storkey_hopfield_diagnostic.py` (`W2_minus_W_frobenius`) | unchanged: reports `||W^2 - W||_F` of the symmetric default |

## Verification

Run before marking ready, and paste the output into the PR:

```bash
pytest tests/test_storkey_hopfield.py -q            # new tests green
git stash push fabricpc/nodes/storkey_hopfield.py   # then rerun: the shape tests must fail on main
pytest -q                                           # whole suite
pre-commit run --all-files
```
