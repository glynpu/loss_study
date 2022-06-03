import torch
from functional import rnnt_loss
# Hypothetical values
logits = torch.tensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.6, 0.1, 0.1],
                         [0.1, 0.1, 0.2, 0.8, 0.1]],
                        [[0.1, 0.6, 0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.2, 0.1, 0.1],
                         [0.7, 0.1, 0.2, 0.1, 0.1]]]],
                      dtype=torch.float32,
                      requires_grad=True)
targets = torch.tensor([[1, 2]], dtype=torch.int)
logit_lengths = torch.tensor([2], dtype=torch.int)
target_lengths = torch.tensor([2], dtype=torch.int)
blank=0
clamp=-1
# transform = transforms.RNNTLoss(blank=0)
loss = rnnt_loss(logits, targets, logit_lengths, target_lengths, blank)
assert torch.isclose(loss, torch.tensor(4.4957)), loss
print(loss)
# loss.backward()
