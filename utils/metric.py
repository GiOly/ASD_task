import torch
import numpy as np
import pandas as pd

from scipy.stats import hmean
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.mixture import GaussianMixture


def decode_class_label(class_label):
    return int(np.where(class_label.detach().cpu().numpy() == 1)[0])


def mahalanobis(x, distribute):
    mu = np.mean(distribute, axis=0, keepdims=True)
    cov = np.cov(distribute.T)
    delta = x - mu
    VI = np.linalg.inv(cov)
    m = np.dot(np.dot(delta, VI), delta.T)
    return m.diagonal()


def represent_extractor(embedding_list, pooling_type, domain_represent=False, var_threshold=False):
    section_embedding_dict = defaultdict(list)
    if domain_represent:
        for emb_dict in embedding_list:
            for e in range(len(emb_dict['embedding'])):
                embedding = emb_dict['embedding'][e]
                class_label = decode_class_label(emb_dict['class_label'][e])  # onehot label convert to num
                domain_label = emb_dict['domain_label'][e]

                class_domain_label = str(class_label) + '_' + domain_label
                section_embedding_dict[class_domain_label].append(embedding)
    else:
        for emb_dict in embedding_list:
            for e in range(len(emb_dict['embedding'])):
                embedding = emb_dict['embedding'][e]
                class_label = decode_class_label(emb_dict['class_label'][e])

                class_label = str(class_label)
                section_embedding_dict[class_label].append(embedding)

    for key, value in section_embedding_dict.items():
        section_embedding_dict[key] = torch.stack(section_embedding_dict[key], dim=0)
        if var_threshold:
            # if var is bigger than threshold, than delete it
            var = torch.var(section_embedding_dict[key], dim=1)
            index = torch.nonzero(var < torch.mean(var).item(), as_tuple=False)
            section_embedding_dict[key] = torch.index_select(section_embedding_dict[key], dim=0, index=index.squeeze())
        if pooling_type == 'avg':
            section_embedding_dict[key] = torch.mean(section_embedding_dict[key], dim=0)
        elif pooling_type == 'lof':
            lof_list = section_embedding_dict[key].squeeze(dim=1).cpu().numpy()
            section_embedding_dict[key] = LocalOutlierFactor(n_neighbors=4, novelty=True).fit(lof_list)
        elif pooling_type == 'knn':
            knn_list = section_embedding_dict[key].squeeze(dim=1).cpu().numpy()
            section_embedding_dict[key] = NearestNeighbors(n_neighbors=1).fit(knn_list)
        elif pooling_type == 'gmm':
            gmm_list = section_embedding_dict[key].squeeze(dim=1).cpu().numpy()
            section_embedding_dict[key] = GaussianMixture(n_components=1, covariance_type='full').fit(gmm_list)
        elif pooling_type == 'nopooling':
            continue
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
            score = min(mahalanobis(embedding.cpu().numpy(), represent_embedding[0].cpu().numpy()),
                        mahalanobis(embedding.cpu().numpy(), represent_embedding[1].cpu().numpy()))
            score = torch.tensor(score, dtype=torch.float32)[0].cuda()
        elif len(represent_embedding) == 1:
            score = mahalanobis(embedding.cpu().numpy(), represent_embedding[0].cpu().numpy())
            score = torch.tensor(score, dtype=torch.float32)[0].cuda()
        else:
            raise NotImplementedError
    elif score_type == 'gmm':
        if len(represent_embedding) == 2:
            score = torch.tensor(min(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()),
                                     -represent_embedding[1].score_samples(embedding.unsqueeze(0).cpu().numpy())), dtype=torch.float32)[0].cuda()
        elif len(represent_embedding) == 1:
            score = torch.tensor(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()), dtype=torch.float32)[0].cuda()
        else:
            raise NotImplementedError
    elif score_type == 'lof':
        if len(represent_embedding) == 2:
            score = torch.tensor(min(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()),
                                     -represent_embedding[1].score_samples(embedding.unsqueeze(0).cpu().numpy())), dtype=torch.float32)[0].cuda()
        elif len(represent_embedding) == 1:
            score = torch.tensor(-represent_embedding[0].score_samples(embedding.unsqueeze(0).cpu().numpy()), dtype=torch.float32)[0].cuda()
        else:
            raise NotImplementedError
    elif score_type == 'knn':
        if len(represent_embedding) == 2:
            score = torch.tensor(min(represent_embedding[0].kneighbors(embedding.unsqueeze(0).cpu().numpy())[0][0],
                                     represent_embedding[1].kneighbors(embedding.unsqueeze(0).cpu().numpy())[0][0]), dtype=torch.float32)[0].cuda()
        elif len(represent_embedding) == 1:
            score = torch.tensor(represent_embedding[0].kneighbors(embedding.unsqueeze(0).cpu().numpy())[0][0], dtype=torch.float32)[0].cuda()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return score


def compute_batch_anomaly_score(detected_embedding_dict, represent_embedding_dict, score_type='cosine', domain_represent=False):
    scores = []
    for e in range(len(detected_embedding_dict['embedding'])):
        embedding = detected_embedding_dict['embedding'][e]
        class_label = decode_class_label(detected_embedding_dict['class_label'][e])

        if domain_represent:
            source_label = str(class_label) + '_' + 'source'
            target_label = str(class_label) + '_' + 'target'
            source_represent_embedding = represent_embedding_dict[source_label]
            target_represent_embedding = represent_embedding_dict[target_label]
            represent_embedding = [source_represent_embedding, target_represent_embedding]
        else:
            represent_embedding = [represent_embedding_dict[str(class_label)]]
        score = anomaly_score_calculator(embedding, represent_embedding, score_type)
        scores.append(score)
    return torch.stack(scores)


def compute_segment_batch_anomaly_score(detected_embedding_dict, represent_embedding_dict, score_type='cosine', domain_represent=False, segment_num=10):
    batch_size = len(detected_embedding_dict['embedding'])
    assert batch_size % segment_num == 0, "validation batch size must be divisible by segment num"
    scores = np.zeros((batch_size // segment_num, segment_num))
    for idx, e in enumerate(range(batch_size)):
        embedding = detected_embedding_dict['embedding'][e] # Embedding of a single sample
        class_label = decode_class_label(detected_embedding_dict['class_label'][e])

        if domain_represent:
            source_label = str(class_label) + '_' + 'source'
            target_label = str(class_label) + '_' + 'target'
            source_represent_embedding = represent_embedding_dict[source_label]
            target_represent_embedding = represent_embedding_dict[target_label]
            represent_embedding = [source_represent_embedding, target_represent_embedding]
        else:
            represent_embedding = [represent_embedding_dict[str(class_label)]]
        score = anomaly_score_calculator(embedding, represent_embedding, score_type)
        scores[idx // segment_num][idx % segment_num] = score
    scores = np.max(scores, axis=1, keepdims=False).tolist()
    return torch.tensor(scores)


def batched_preds(anomaly_scores, class_labels, anomaly_labels, domain_labels, filenames, lable_dict):
    prediction_df = pd.DataFrame()

    for i in range(anomaly_scores.shape[0]):
        anomaly_score = anomaly_scores[i].item()
        class_label = decode_class_label(class_labels[i])
        anomaly_label = anomaly_labels[i].item()
        domain_label = domain_labels[i]
        filename = filenames[i]

        machine_label, section_label = lable_dict[class_label].split('/')
        result_labels = [[filename, anomaly_score, anomaly_label, machine_label, section_label, domain_label]]
        prediction_df = prediction_df.append(result_labels, ignore_index=True)
    return prediction_df


def compute_test_auc(prediction_df, max_fpr=0.1):
    machine_set = sorted(list(set(prediction_df["machine_label"])))
    section_set = sorted(set(prediction_df["section_label"]))
    result = {}
    for machine in machine_set:
        machine_result_df = pd.DataFrame()
        for section in section_set:
            source_df = prediction_df[(prediction_df['machine_label'] == machine) &
                                      (prediction_df['section_label'] == section) &
                                      (prediction_df['domain_label'] == 'source')]
            target_df = prediction_df[(prediction_df['machine_label'] == machine) &
                                      (prediction_df['section_label'] == section) &
                                      (prediction_df['domain_label'] == 'target')]
            df = pd.concat([source_df, target_df])
            auc = roc_auc_score(df.anomaly_label, df.anomaly_score)
            pauc = roc_auc_score(df.anomaly_label, df.anomaly_score, max_fpr=max_fpr)
            source_auc = roc_auc_score(source_df.anomaly_label, source_df.anomaly_score)
            source_pauc = roc_auc_score(source_df.anomaly_label, source_df.anomaly_score, max_fpr=max_fpr)
            target_auc = roc_auc_score(target_df.anomaly_label, target_df.anomaly_score)
            target_pauc = roc_auc_score(target_df.anomaly_label, target_df.anomaly_score, max_fpr=max_fpr)
            machine_result_df = machine_result_df.append([[section, auc, pauc, source_auc, source_pauc, target_auc, target_pauc]], ignore_index=True)
        mean = []
        har_mean = []
        for index, col in machine_result_df.iteritems():
            if index == 0:
                continue
            mean.append(np.mean(machine_result_df[index]))
            har_mean.append(hmean(machine_result_df[index]))
        machine_result_df = machine_result_df.append([["mean"] + mean], ignore_index=True)
        machine_result_df = machine_result_df.append([["hmean"] + har_mean], ignore_index=True)

        machine_result_df.columns = ['section', 'auc', 'pauc', 'source_auc', 'source_pauc', 'target_auc', 'target_pauc']
        machine_result_df.reset_index(drop=True)
        result[machine] = machine_result_df

    return result


def compute_segment_test_auc(prediction_df, max_fpr=0.1):
    machine_set = sorted(list(set(prediction_df["machine_label"])))
    section_set = sorted(set(prediction_df["section_label"]))
    result = {}
    for machine in machine_set:
        machine_result_df = pd.DataFrame()
        for section in section_set:
            source_df = prediction_df[(prediction_df['machine_label'] == machine) &
                                      (prediction_df['section_label'] == section) &
                                      (prediction_df['domain_label'] == 'source')]
            target_df = prediction_df[(prediction_df['machine_label'] == machine) &
                                      (prediction_df['section_label'] == section) &
                                      (prediction_df['domain_label'] == 'target')]
            df = pd.concat([source_df, target_df])
            auc = roc_auc_score(df.anomaly_label, df.anomaly_score)
            pauc = roc_auc_score(df.anomaly_label, df.anomaly_score, max_fpr=max_fpr)
            source_auc = roc_auc_score(source_df.anomaly_label, source_df.anomaly_score)
            source_pauc = roc_auc_score(source_df.anomaly_label, source_df.anomaly_score, max_fpr=max_fpr)
            target_auc = roc_auc_score(target_df.anomaly_label, target_df.anomaly_score)
            target_pauc = roc_auc_score(target_df.anomaly_label, target_df.anomaly_score, max_fpr=max_fpr)
            machine_result_df = machine_result_df.append([[section, auc, pauc, source_auc, source_pauc, target_auc, target_pauc]], ignore_index=True)
        mean = []
        har_mean = []
        for index, col in machine_result_df.iteritems():
            if index == 0:
                continue
            mean.append(np.mean(machine_result_df[index]))
            har_mean.append(hmean(machine_result_df[index]))
        machine_result_df = machine_result_df.append([["mean"] + mean], ignore_index=True)
        machine_result_df = machine_result_df.append([["hmean"] + har_mean], ignore_index=True)

        machine_result_df.columns = ['section', 'auc', 'pauc', 'source_auc', 'source_pauc', 'target_auc', 'target_pauc']
        machine_result_df.reset_index(drop=True)
        result[machine] = machine_result_df

    return result
