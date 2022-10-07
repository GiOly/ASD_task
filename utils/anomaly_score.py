import torch
import numpy as np


def mahalanobis(x, distribute):
    mu = np.mean(distribute, axis=0, keepdims=True)
    cov = np.cov(distribute.T)
    delta = x - mu
    VI = np.linalg.inv(cov)
    m = np.dot(np.dot(delta, VI), delta.T)
    return np.sqrt(m)


def anomaly_score_calculator(embedding, represent_embedding, score_type):
    if score_type == 'cosine':
        if len(represent_embedding) == 2:
            score = min(1 - torch.cosine_similarity(embedding, represent_embedding[0], dim=0),
                        1 - torch.cosine_similarity(embedding, represent_embedding[1], dim=0))
        elif len(represent_embedding) == 1:
            score = 1 - torch.cosine_similarity(embedding, represent_embedding[0], dim=0)
        else:
            raise NotImplementedError

    elif score_type == 'mahalanobis':
        if len(represent_embedding) == 2:
            score = min(mahalanobis(embedding.cpu().numpy(), represent_embedding[0].squeeze(1).cpu().numpy()),
                        mahalanobis(embedding.cpu().numpy(), represent_embedding[1].squeeze(1).cpu().numpy()))
        elif len(represent_embedding) == 1:
            score = mahalanobis(embedding.cpu().numpy(), represent_embedding[0].squeeze(1).cpu().numpy())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return score
