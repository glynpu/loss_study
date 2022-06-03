# copied from
# https://pytorch.org/audio/0.10.0/transforms.html#rnntloss
import torch
import torchaudio

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
transform = torchaudio.transforms.RNNTLoss(blank=0)
loss = transform(logits, targets, logit_lengths, target_lengths)
loss.backward()
print(loss)

