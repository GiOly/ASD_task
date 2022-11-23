import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torchaudio


def pad_audio(audio, target_len):
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )
    else:
        audio = audio[:target_len]
    return audio


def read_audio(file, pad_to):
    audio, sr = torchaudio.load(file)
    if pad_to is not None:
        audio = pad_audio(audio, pad_to)
    return audio


class ASDDataset(Dataset):
    def __init__(self,
                 audio_folder,
                 csv_entries,
                 class_label_dict,
                 pad_to=10,
                 fs=16000,
                 dir_name='train',
                 feats_pipeline=None,
                 return_class_label=False,
                 return_anomaly_label=False,
                 return_domain_label=False,
                 return_machine_label=False,
                 return_filename=False):
        self.fs = fs
        self.pad_to = pad_to * fs
        self.label_dict = class_label_dict
        self.feats_pipeline = feats_pipeline
        self.return_class_label = return_class_label
        self.return_anomaly_label = return_anomaly_label
        self.return_domain_label = return_domain_label
        self.return_machine_label = return_machine_label
        self.return_filename = return_filename

        examples = {}
        for i, r in csv_entries.iterrows():
            if int(r["section"].split('_')[1]) in [0, 1, 2]:
                folder = os.path.join(audio_folder, 'dev_data')
            else:
                folder = os.path.join(audio_folder, 'eval_data')
            if r['filename'] not in examples.keys():
                examples[r["filename"]] = {
                    "audio_path": os.path.join(folder, r["machine"], dir_name, r["filename"]),
                    "machine": r["machine"],
                    "section": r["section"],
                    "domain": r['domain'],
                    "anomaly": r['label'],
                    "attribute": self.split_attribute(r["attribute"])
                }
        self.examples = examples
        self.examples_list = list(self.examples.keys())

    def split_attribute(self, attribute_string):
        att_list = attribute_string.split('_')
        assert len(att_list) % 2 == 0, \
            "This attribute string cannot separate out a suitable key-value pair"
        att_dict = {}
        for k, v in zip(att_list[::2], att_list[1::2]):
            att_dict[k] = v
        return att_dict

    def encode_class_label(self, example):
        label_idx = self.label_dict[example['machine'] + '/' + example['section']]
        y = np.zeros(len(self.label_dict.keys()))
        y[label_idx] = 1
        return y

    def encode_anomaly_label(self, example):
        if example['anomaly'] == 'normal':
            return 0
        else:
            return 1

    def __getitem__(self, item):
        file = self.examples_list[item]
        example = self.examples[file]

        audio = read_audio(example['audio_path'], self.pad_to)
        out_args = [audio]
        if self.return_class_label:
            class_label = self.encode_class_label(example)
            out_args.append(class_label)
        if self.return_anomaly_label:
            anomaly_label = self.encode_anomaly_label(example)
            out_args.append(anomaly_label)
        if self.return_domain_label:
            domain_label = example["domain"]
            out_args.append(domain_label)
        if self.return_machine_label:
            machine_label = example["machine"]
            out_args.append(machine_label)
        if self.return_filename:
            out_args.append(example['audio_path'])

        return out_args

    def __len__(self):
        return len(self.examples_list)


class ASDDataset_segment(Dataset):
    def __init__(self,
                 audio_folder,
                 csv_entries,
                 class_label_dict,
                 pad_to=10,
                 fs=16000,
                 segment_len=1,
                 dir_name='train',
                 feats_pipeline=None,
                 return_class_label=False,
                 return_anomaly_label=False,
                 return_domain_label=False,
                 return_machine_label=False,
                 return_filename=False,
                 ):
        self.fs = fs
        self.pad_to = pad_to * fs
        self.segment_len = segment_len * fs
        self.segment_num = self.pad_to // self.segment_len
        self.label_dict = class_label_dict
        self.feats_pipeline = feats_pipeline
        self.return_class_label = return_class_label
        self.return_anomaly_label = return_anomaly_label
        self.return_domain_label = return_domain_label
        self.return_machine_label = return_machine_label
        self.return_filename = return_filename
        csv_entries = pd.DataFrame(np.repeat(csv_entries.values, self.segment_num, axis=0), columns=csv_entries.columns)
        examples = {}
        for i, r in csv_entries.iterrows():
            segment_idx = i % self.segment_num
            if int(r["section"].split('_')[1]) in [0, 1, 2]:
                folder = os.path.join(audio_folder, 'dev_data')
            else:
                folder = os.path.join(audio_folder, 'eval_data')
            key = r['filename'] + '_' + str(segment_idx)
            if key not in examples.keys():
                examples[key] = {
                    "audio_path": os.path.join(folder, r["machine"], dir_name, r["filename"]),
                    "segment_idx": i % self.segment_num,
                    "machine": r["machine"],
                    "section": r["section"],
                    "domain": r['domain'],
                    "anomaly": r['label'],
                    "attribute": self.split_attribute(r["attribute"])
                }
        self.examples = examples
        self.examples_list = list(self.examples.keys())

    def split_attribute(self, attribute_string):
        att_list = attribute_string.split('_')
        assert len(att_list) % 2 == 0, \
            "This attribute string cannot separate out a suitable key-value pair"
        att_dict = {}
        for k, v in zip(att_list[::2], att_list[1::2]):
            att_dict[k] = v
        return att_dict

    def encode_class_label(self, example):
        label_idx = self.label_dict[example['machine'] + '/' + example['section']]
        y = np.zeros(len(self.label_dict.keys()))
        y[label_idx] = 1
        return y

    def encode_anomaly_label(self, example):
        if example['anomaly'] == 'normal':
            return 0
        else:
            return 1

    def __getitem__(self, item):
        file = self.examples_list[item]
        example = self.examples[file]

        audio = read_audio(example['audio_path'], self.pad_to)
        segment_idx = example['segment_idx']
        audio = audio[:, segment_idx * self.segment_len: (segment_idx + 1) * self.segment_len]
        out_args = [audio]
        if self.return_class_label:
            class_label = self.encode_class_label(example)
            out_args.append(class_label)
        if self.return_anomaly_label:
            anomaly_label = self.encode_anomaly_label(example)
            out_args.append(anomaly_label)
        if self.return_domain_label:
            domain_label = example["domain"]
            out_args.append(domain_label)
        if self.return_machine_label:
            machine_label = example["machine"]
            out_args.append(machine_label)
        if self.return_filename:
            out_args.append(example['audio_path'])

        return out_args

    def __len__(self):
        return len(self.examples_list)
