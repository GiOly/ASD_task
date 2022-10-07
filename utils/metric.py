import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import hmean
import numpy as np
import torch
from utils.anomaly_score import anomaly_score_calculator


def compute_batch_anomaly_score(detected_embedding_dict, represent_embedding_dict, score_type='cosine'):
    scores = []
    for e in range(len(detected_embedding_dict['embedding'])):
        embedding = detected_embedding_dict['embedding'][e]
        class_label = detected_embedding_dict['class_label'][e].item()

        if detected_embedding_dict['domain_label'] is not None:
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


def batched_preds(anomaly_scores, class_labels, anomaly_labels, domain_labels, filenames, lable_dict):
    prediction_df = pd.DataFrame()

    for i in range(anomaly_scores.shape[0]):
        anomaly_score = anomaly_scores[i].item()
        class_label = class_labels[i].item()
        anomaly_label = anomaly_labels[i].item()
        domain_label = domain_labels[i]
        filename = filenames[i]

        machine_label, section_label = lable_dict[class_label].split('/')
        result_labels = [[filename, anomaly_score, anomaly_label, machine_label, section_label, domain_label]]
        curr_pred = pd.DataFrame(result_labels)
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