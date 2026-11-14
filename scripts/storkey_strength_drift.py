"""Track StorkeyHopfield's learnable strength over a long PC training run.

Linear -> StorkeyHopfield -> softmax classifier on synthetic Gaussian clusters.
Prints the effective strength s, the training energy, and the latent-Hessian
floor 1 + (s / D) * min eig(W^2 - W), which is negative once s > 4 D.

    python scripts/storkey_strength_drift.py --steps 14000
    python scripts/storkey_strength_drift.py --steps 4000 --stop-grad
    python scripts/storkey_strength_drift.py --steps 14000 --cap 100
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode, Linear, StorkeyHopfield
from fabricpc.nodes import storkey_hopfield
from fabricpc.nodes.base import NodeBase
from fabricpc.training import make_train_step

D, CLASSES, IN, BATCH = 32, 4, 64, 64


def freeze_strength_in_hopfield_term():
    """Treat s as a constant inside E_hop, so it is learned through z_mu only."""

    def energy(params, inputs, state, aux, node_info):
        base = NodeBase.energy(params, inputs, state, aux, node_info)
        if aux is None:
            return base
        W, strength = aux
        z = state.z_latent
        wz = z @ W
        e_hop = (0.5 / z.shape[-1]) * jnp.sum(wz * (wz - z), axis=-1)
        return base + jax.lax.stop_gradient(strength) * e_hop

    storkey_hopfield.StorkeyHopfield.energy = staticmethod(energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--stop-grad", action="store_true")
    ap.add_argument("--cap", type=float, default=None)
    args = ap.parse_args()
    if args.stop_grad:
        freeze_strength_in_hopfield_term()

    x_node = IdentityNode(shape=(IN,), name="input")
    hidden = Linear(shape=(D,), activation=SigmoidActivation(), name="hidden")
    hop = StorkeyHopfield(shape=(D,), name="hopfield", max_strength=args.cap)
    out = Linear(
        shape=(CLASSES,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
    )
    structure = graph(
        nodes=[x_node, hidden, hop, out],
        edges=[
            Edge(source=x_node, target=hidden.slot("in")),
            Edge(source=hidden, target=hop.slot("in")),
            Edge(source=hop, target=out.slot("in")),
        ],
        task_map=TaskMap(x=x_node, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )

    key = jax.random.PRNGKey(0)
    params = initialize_params(structure, key)
    centers = jax.random.normal(jax.random.PRNGKey(1), (CLASSES, IN))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    step = make_train_step(structure, optimizer)
    edge = next(iter(params.nodes["hopfield"].weights))

    every = max(1, args.steps // 8)
    for i in range(args.steps + 1):
        key, k_y, k_x, k_step = jax.random.split(key, 4)
        y = jax.random.randint(k_y, (BATCH,), 0, CLASSES)
        x = centers[y] + 0.8 * jax.random.normal(k_x, (BATCH, IN))
        batch = {"x": x, "y": jax.nn.one_hot(y, CLASSES)}
        params, opt_state, metrics, _ = step(params, opt_state, batch, k_step)
        if i % every == 0:
            cfg = {"hopfield_strength": None, "max_strength": args.cap}
            node_params = params.nodes["hopfield"]
            s = float(StorkeyHopfield.effective_strength(node_params, cfg))
            W = np.asarray(params.nodes["hopfield"].weights[edge])
            eig = np.linalg.eigvalsh(0.5 * (W + W.T))
            floor = 1 + s / D * (eig**2 - eig).min()
            print(
                f"step {i:6d}  energy {float(metrics['energy']):9.4f}  "
                f"s {s:8.3f}  4D {4 * D}  hessian floor {floor:7.3f}"
            )


if __name__ == "__main__":
    main()
