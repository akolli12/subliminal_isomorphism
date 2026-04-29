"""
Post-hoc evaluation:
  - phi_inv exact-match accuracy on the training set
  - phi_inv exact-match accuracy on a freshly sampled held-out set
    (tests whether the model generalizes the inversion function or
    just memorized the 100 training examples)
  - Sample (phi, psi target/argmax, phi_inv target/argmax) rows
"""

import torch

import config
from model import TinyTransformer
from data.generate_dataset import load_dataset, build_sequence, sample_online

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(path=config.MODEL_PATH):
    model = TinyTransformer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def phi_inv_accuracy(model, sequences):
    pred   = model(sequences)[:, config.PHI_INV_LOGITS, :].argmax(dim=-1)
    target = sequences[:, config.PHI_INV]
    correct = (pred == target).all(dim=1).sum().item()
    return correct, sequences.shape[0]


@torch.no_grad()
def show_examples(model, sequences, k=5):
    x = sequences[:k]
    logits = model(x)

    phi          = x[:, config.PHI].cpu()
    psi          = x[:, config.PSI].cpu()
    phi_inv      = x[:, config.PHI_INV].cpu()
    psi_pred     = logits[:, config.PSI_LOGITS, :].argmax(dim=-1).cpu()
    phi_inv_pred = logits[:, config.PHI_INV_LOGITS, :].argmax(dim=-1).cpu()

    print("\n" + "=" * 60)
    print("Sample model outputs (values are 0-indexed)")
    print("=" * 60)
    for i in range(k):
        ok = "OK" if (phi_inv_pred[i] == phi_inv[i]).all() else "WRONG"
        print(f"\n--- example {i} ---")
        print(f"  phi (input)            : {phi[i].tolist()}")
        print(f"  phi_inv target         : {phi_inv[i].tolist()}")
        print(f"  phi_inv argmax         : {phi_inv_pred[i].tolist()}  [{ok}]")
        print(f"  psi target (random)    : {psi[i].tolist()}")
        print(f"  psi argmax             : {psi_pred[i].tolist()}")


def main():
    model = load_model()

    # --- training set accuracy ---
    phi, psi = load_dataset()
    train_seq = build_sequence(phi, psi).to(DEVICE)
    correct, total = phi_inv_accuracy(model, train_seq)
    print(f"phi_inv accuracy on TRAIN set: {correct}/{total} "
          f"({100 * correct / total:.1f}%)")

    # --- held-out set accuracy (fresh phis) ---
    held_phi, held_psi = sample_online(batch_size=512)
    held_seq = build_sequence(held_phi, held_psi).to(DEVICE)
    correct, total = phi_inv_accuracy(model, held_seq)
    print(f"phi_inv accuracy on HELD-OUT set: {correct}/{total} "
          f"({100 * correct / total:.1f}%)")

    show_examples(model, train_seq)


if __name__ == "__main__":
    main()