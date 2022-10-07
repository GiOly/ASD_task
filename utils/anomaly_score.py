import torch
import numpy as np
from collections import defaultdict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture


def mahalanobis(x, distribute):
    mu = np.mean(distribute, axis=0, keepdims=True)
    cov = np.cov(distribute.T)
    delta = x - mu
    VI = np.linalg.inv(cov)
    m = np.dot(np.dot(delta, VI), delta.T)
    return np.sqrt(m)


def represent_extractor(embedding_list, pooling_type, domain_represent=False):
    section_embedding_dict = defaultdict(list)
    if domain_represent:
        for emb_dict in embedding_list:
            for e in range(len(emb_dict['embedding'])):
                embedding = emb_dict['embedding'][e]
                class_label = emb_dict['class_label'][e].item()
                domain_label = emb_dict['domain_label'][e]

                class_domain_label = str(class_label) + '_' + domain_label
                section_embedding_dict[class_domain_label].append(embedding)
    else:
        for emb_dict in embedding_list:
            for e in range(len(emb_dict['embedding'])):
                embedding = emb_dict['embedding'][e]
                class_label = emb_dict['class_label'][e].item()

                class_label = str(class_label)
                section_embedding_dict[class_label].append(embedding)

    for key, value in section_embedding_dict.items():
        section_embedding_dict[key] = torch.stack(section_embedding_dict[key], dim=0)
        if pooling_type == 'avg':
            section_embedding_dict[key] = torch.mean(section_embedding_dict[key], dim=0)
        elif pooling_type == 'LOF':
            lof_list = section_embedding_dict[key].squeeze(dim=1).cpu().numpy()
            section_embedding_dict[key] = LocalOutlierFactor(n_neighbors=4, novelty=True).fit(lof_list)
        elif pooling_type == 'GMM':
            gmm_list = section_embedding_dict[key].squeeze(dim=1).cpu().numpy()
            section_embedding_dict[key] = GaussianMixture(n_components=5, covariance_type='full').fit(gmm_list)
        else:
            raise NotImplementedError

    return section_embedding_dict


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
    elif score_type == 'GMM':
        if len(represent_embedding) == 2:
            score = torch.tensor(min(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()),
                                     -represent_embedding[1].score_samples(embedding.unsqueeze(0).cpu().numpy())), dtype=torch.float32)[0].cuda()
        elif len(represent_embedding) == 1:
            score = torch.tensor(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()), dtype=torch.float32)[0].cuda()
        else:
            raise NotImplementedError
    elif score_type == 'LOF':
        if len(represent_embedding) == 2:
            score = torch.tensor(min(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()),
                                     -represent_embedding[1].score_samples(embedding.unsqueeze(0).cpu().numpy())), dtype=torch.float32)[0].cuda()
        elif len(represent_embedding) == 1:
            score = torch.tensor(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()), dtype=torch.float32)[0].cuda()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return score
