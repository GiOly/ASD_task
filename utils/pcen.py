import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class F2M(nn.Module):
    """This turns a normal STFT into a MEL Frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.
    Args:
        n_mels (int): number of MEL bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: sr // 2
        f_min (float): minimum frequency. default: 0
    """

    def __init__(self, n_mels=40, sr=16000, f_max=None, f_min=0., n_fft=40, onesided=True):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        if onesided:
            self.n_fft = self.n_fft // 2 + 1
        self._init_buffers()

    def _init_buffers(self):
        m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
        m_max = 2595 * np.log10(1. + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = (700 * (10 ** (m_pts / 2595) - 1))

        bins = torch.floor(((self.n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(self.n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
        self.register_buffer("fb", fb)

    def forward(self, spec_f):
        spec_m = torch.matmul(spec_f, self.fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m


def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False, last_state=None, empty=True):
    frames = x.split(1, -2)
    m_frames = []
    if empty:
        last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_, last_state


class StreamingPCENTransform(nn.Module):

    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=False,
                 use_cuda_kernel=False, **stft_kwargs):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        if trainable:
            self.s = nn.Parameter(torch.Tensor([s]))
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
            self.delta = nn.Parameter(torch.Tensor([delta]))
            self.r = nn.Parameter(torch.Tensor([r]))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable
        self.stft_kwargs = stft_kwargs
        self.register_buffer("last_state", torch.zeros(stft_kwargs["n_mels"]))
        mel_keys = {"n_mels", "sr", "f_max", "f_min", "n_fft"}
        mel_keys = set(stft_kwargs.keys()).intersection(mel_keys)
        mel_kwargs = {k: stft_kwargs[k] for k in mel_keys}
        stft_keys = set(stft_kwargs.keys()) - mel_keys
        self.n_fft = stft_kwargs["n_fft"]
        self.stft_kwargs = {k: stft_kwargs[k] for k in stft_keys}
        self.f2m = F2M(**mel_kwargs)
        self.reset()

    def reset(self):
        self.empty = True

    def forward(self, x):
        x = torch.stft(x, self.n_fft, **self.stft_kwargs).norm(dim=-1, p=2)
        x = self.f2m(x.permute(0, 2, 1))
        x, ls = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable, self.last_state, self.empty)
        self.last_state = ls.detach()
        self.empty = False
        return x
