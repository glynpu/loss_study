import torch
from pathlib import Path
from torch import Tensor
def _load_lib() -> bool:
    path = Path("./build/libtorchaudio.so")
    if not path.exists():
        return False
    torch.ops.load_library(path)
    torch.classes.load_library(path)
    return True

_load_lib()

def rnnt_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = -1,
    clamp: float = -1,
    reduction: str = "mean",
):
    """Compute the RNN Transducer loss from *Sequence Transduction with Recurrent Neural Networks*
    [:footcite:`graves2012sequence`].
    .. devices:: CPU CUDA
    .. properties:: Autograd TorchScript
    The RNN Transducer loss extends the CTC loss by defining a distribution over output
    sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.
    Args:
        logits (Tensor): Tensor of dimension `(batch, max seq length, max target length + 1, class)`
            containing output from joiner
        targets (Tensor): Tensor of dimension `(batch, max target length)` containing targets with zero padded
        logit_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of each sequence from encoder
        target_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of targets for each sequence
        blank (int, optional): blank label (Default: ``-1``)
        clamp (float, optional): clamp for gradients (Default: ``-1``)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. (Default: ``'mean'``)
    Returns:
        Tensor: Loss with the reduction option applied. If ``reduction`` is  ``'none'``, then size `(batch)`,
        otherwise scalar.
    """
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError("reduction should be one of 'none', 'mean', or 'sum'")

    if blank < 0:  # reinterpret blank index if blank < 0.
        blank = logits.shape[-1] + blank

    costs, _ = torch.ops.torchaudio.rnnt_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=blank,
        clamp=clamp,
    )

    if reduction == "mean":
        return costs.mean()
    elif reduction == "sum":
        return costs.sum()

    return costs

a = torch.tensor([0.0])
rnnt_loss(a, a, a, a, 0, 0.0)
