import torch
import numpy as np


def mixup(features, label=None, permutation=None, c=None, alpha=0.2, beta=0.2, mixup_label_type="soft", returnc=False):
    with torch.no_grad():
        batch_size = features.size(0)

        if permutation is None:
            permutation = torch.randperm(batch_size)

        if c is None:
            if mixup_label_type == "soft":
                c = np.random.beta(alpha, beta)
            elif mixup_label_type == "hard":
                c = np.random.beta(alpha, beta) * 0.4 + 0.3  # c in [0.3, 0.7]

        mixed_features = c * features + (1 - c) * features[permutation, :]
        if label is not None:
            if mixup_label_type == "soft":
                mixed_label = torch.clamp(c * label + (1 - c) * label[permutation, :], min=0, max=1)
            elif mixup_label_type == "hard":
                mixed_label = torch.clamp(label + label[permutation, :], min=0, max=1)
            else:
                raise NotImplementedError(f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                                          f"{'soft', 'hard'}")
            if returnc:
                return mixed_features, mixed_label, c, permutation
            else:
                return mixed_features, mixed_label
        else:
            return mixed_features

