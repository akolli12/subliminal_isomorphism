"""Dataset construction and baseline/simulator-aligned training."""

import os

import torch
import torch.nn.functional as F

from data_generation.data import graphs, permutations, permute_graph
from src.config import set_seed
from src.layouts import (IGNORE, PermutationContext, build_batch, graph_layout,
                         graph_prefix, graph_seq_len, perm_layout, perm_seq_len,
                         specials)
from src.model import TinyTransformer
from src.sampling import sample_psi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataset(config, graph_conditioned, simulator_aligned,
                 reset_global_seed=True):
    """Match the original per-n generation order exactly."""
    # Legacy behavior: the three helper-based trainers reset the global RNG
    # here; graph_train_soft used only a local data generator and did not.
    if reset_global_seed:
        set_seed(config.data_seed)
    generator = torch.Generator().manual_seed(config.data_seed)
    ns, graph_rows, phis, psis = [], [], [], []
    for n in config.ns:
        num_examples = config.counts[n]
        if graph_conditioned:
            graph_rows += list(graphs(num_examples, n, generator))
        phis += list(permutations(num_examples, n, generator))
        psis += list(permutations(num_examples, n, generator))
        ns += [n] * num_examples
    return build_batch(
        ns, phis, psis, config.max_n,
        graph_rows if graph_conditioned else None,
        soft=simulator_aligned,
        uniform_psi=simulator_aligned,
    )


def train(config, graph_conditioned, simulator_aligned, seed, checkpoint,
          preserve_training_seed=False):
    """Train one shared model; operation order matches the current implementation."""
    set_seed(seed)
    dataset_tensors = make_dataset(
        config, graph_conditioned, simulator_aligned,
        reset_global_seed=not preserve_training_seed,
    )
    dataset_tensors = tuple(tensor.to(DEVICE) for tensor in dataset_tensors)
    tokens = dataset_tensors[0]
    model = TinyTransformer(
        specials(config.max_n)["vocab"],
        graph_seq_len(config.max_n) if graph_conditioned else perm_seq_len(config.max_n),
        config.d_model, config.n_heads, config.n_layers,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for step in range(config.steps):
        indices = torch.randint(0, len(tokens), (config.batch_size,), device=DEVICE)
        logits = model(tokens[indices])
        if simulator_aligned:
            targets, masks = dataset_tensors[1], dataset_tensors[2]
            loss = (-(targets[indices] * F.log_softmax(logits, -1)).sum(-1))[masks[indices]].mean()
        else:
            labels = dataset_tensors[1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels[indices].reshape(-1), ignore_index=IGNORE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 2000 == 0 or step == config.steps - 1:
            print(f"  step {step:6d} loss={loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model


def _valid_rate(model, contexts, layout):
    samples = sample_psi(model, contexts, layout, valid=False)
    expected = torch.arange(layout.n, device=samples.device)
    return (samples.sort(1).values == expected).all(1).float().mean().item()


def _block_accuracy(model, sequences, layout, name):
    block = layout[name]
    logits = model(sequences.to(DEVICE))
    predicted = logits[:, block.start - 1:block.stop - 1, :layout.n].argmax(-1)
    return (predicted == sequences[:, block].to(DEVICE)).all(1).float().mean().item()


@torch.no_grad()
def diagnostics(model, config, graph_conditioned, seed):
    """Run the exact functionality diagnostics and preserve their draw order."""
    generator = torch.Generator().manual_seed(seed + 100)
    output = {}
    for n in config.ns:
        if graph_conditioned:
            layout = graph_layout(n, config.max_n)
            context_phis = permutations(1000, n, generator)
            context_g0 = graphs(1000, n, torch.Generator().manual_seed(seed))
            contexts = graph_prefix(
                context_g0, permute_graph(context_g0, context_phis, n),
                context_phis, config.max_n,
            )
            phis = permutations(1000, n, generator)
            psis = permutations(1000, n, generator)
            graph_rows = graphs(1000, n, generator)
            sequences, _ = build_batch([n] * 1000, list(phis), list(psis),
                                       config.max_n, list(graph_rows))
        else:
            layout = perm_layout(n, config.max_n)
            contexts = PermutationContext(config.max_n)(permutations(2000, n, generator))
            phis = permutations(1000, n, generator)
            psis = permutations(1000, n, generator)
            sequences, _ = build_batch([n] * 1000, list(phis), list(psis), config.max_n)
        row = {
            "psi_valid_pct": 100 * _valid_rate(model, contexts, layout),
            "psi_inv_acc_pct": 100 * _block_accuracy(model, sequences, layout, "psi_inv"),
        }
        if not graph_conditioned:
            row["phi_psi_inv_acc_pct"] = 100 * _block_accuracy(
                model, sequences, layout, "phi_psi_inv"
            )
        output[n] = row
        print(f"  n={n}: {row}", flush=True)
    return output
