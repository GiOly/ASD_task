import torch
import numpy as np
import random


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


def add_noise(features, snrs=(30, 40), dims=(1, 2)):
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand((features.shape[0],), device=features.device).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)
    sigma = torch.std(features, dim=dims, keepdim=True) / snr
    return features + torch.randn(features.shape, device=features.device) * sigma


def frame_shift(features, label=None, net_pooling=None):
    if label is not None:
        batch_size, _, _ = features.shape
        shifted_feature = []
        shifted_label = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
            shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
            shifted_label.append(torch.roll(label[idx], shift, dims=-1))
        return torch.stack(shifted_feature), torch.stack(shifted_label)
    else:
        batch_size, _, _ = features.shape
        shifted_feature = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
        return torch.stack(shifted_feature)
