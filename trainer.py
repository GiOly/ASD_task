import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from collections import defaultdict
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchmetrics import Accuracy, AUROC
from utils.scaler import TorchScaler
from utils.anomaly_score import anomaly_score_calculator
from utils.metric import batched_preds, compute_test_auc, compute_batch_anomaly_score
from tqdm import tqdm
import csv


class ASDTask(pl.LightningModule):
    def __init__(
            self,
            hparams,
            model,
            opt,
            train_data,
            valid_data,
            test_data,
            scheduler,
            fast_dev_run=False,
    ):
        super(ASDTask, self).__init__()

        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.test_data_label_dict = {v: k for k, v in test_data.label_dict.items()}
        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        self.scaler = self._init_scaler()

        self.supervised_loss = torch.nn.CrossEntropyLoss()

        self.embedding_list = []

        self.accuracy_calculator = Accuracy()
        self.auc_calculator = AUROC()
        self.pauc_calculator = AUROC(max_fpr=self.hparams["training"]["max_fpr"])

        self.test_buffer = pd.DataFrame()

    def _init_scaler(self):
        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.configs["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, label, model):
        return model(self.scaler(self.take_log(mel_feats)), label)

    def training_step(self, batch):
        if self.hparams["represent"]["domain_represent"]:
            audio, class_labels, domain_labels = batch
        else:
            audio, class_labels = batch
            domain_labels = None
        feature = self.mel_spec(audio)
        preds, _ = self.detect(feature, class_labels, self.model)

        loss = self.supervised_loss(preds, class_labels)

        self.log('train/step', self.scheduler["scheduler"]._step_count, prog_bar=True)
        self.log('train/loss', loss)
        self.log('train/lr', self.opt.param_groups[-1]["lr"])

        return loss

    def represent_extractor(self, pooling_type, domain_represent=False):

        section_embedding_dict = defaultdict(list)
        if domain_represent:
            for emb_dict in self.embedding_list:
                for e in range(len(emb_dict['embedding'])):
                    embedding = emb_dict['embedding'][e]
                    class_label = emb_dict['class_label'][e].item()
                    domain_label = emb_dict['domain_label'][e]

                    class_domain_label = str(class_label) + '_' + domain_label
                    section_embedding_dict[class_domain_label].append(embedding)
        else:
            for emb_dict in self.embedding_list:
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
                pass
            elif pooling_type == 'GMM':
                pass

        return section_embedding_dict

    def on_validation_epoch_start(self):
        self.embedding_list = []
        for batch in self.train_dataloader():

            if self.hparams["represent"]["domain_represent"]:
                audio, class_labels, domain_labels = batch
            else:
                audio, class_labels = batch
                domain_labels = None
            mels = self.mel_spec(audio.cuda())
            self.model.eval()
            with torch.no_grad():
                _, embedding = self.detect(mels, class_labels.cuda(), self.model)

            self.embedding_list.append({'embedding': embedding,
                                        'domain_label': domain_labels,
                                        'class_label': class_labels})
        self.represent_embedding_dict = self.represent_extractor(pooling_type=self.hparams["represent"]["pooling_type"],
                                                                 domain_represent=self.hparams["represent"]["domain_represent"])

    def validation_step(self, batch, batch_indx):
        if self.hparams["represent"]["domain_represent"]:
            audio, class_labels, anomaly_labels, domain_labels = batch
        else:
            audio, class_labels, anomaly_labels = batch
            domain_labels = None

        mels = self.mel_spec(audio)
        preds, embedding = self.detect(mels, class_labels, self.model)

        valid_loss = self.supervised_loss(preds, class_labels)

        detected_embedding_dict = {
            'embedding': embedding,
            'domain_label': domain_labels,
            'class_label': class_labels
        }

        anomaly_score = compute_batch_anomaly_score(detected_embedding_dict,
                                                    self.represent_embedding_dict,
                                                    self.hparams["represent"]["score_type"])

        self.accuracy_calculator.update(preds, class_labels)
        self.auc_calculator.update(anomaly_score, anomaly_labels)
        self.pauc_calculator.update(anomaly_score, anomaly_labels)

        return

    def validation_epoch_end(self, outputs):

        accuracy = self.accuracy_calculator.compute()
        auc = self.auc_calculator.compute()
        pauc = self.pauc_calculator.compute()
        overall = auc + pauc

        self.log('valid/accuracy', accuracy)
        self.log('valid/overall', overall, prog_bar=True)
        self.log('valid/auc', auc)
        self.log('valid/pauc', pauc)
        return overall

    def on_save_checkpoint(self, checkpoint):
        checkpoint = self.model.state_dict()
        return checkpoint

    def on_test_epoch_start(self):
        self.embedding_list = []
        self.train_data.return_domain_label = True
        for batch in self.train_dataloader():
            audio, class_labels, domain_labels = batch
            mels = self.mel_spec(audio.cuda())
            self.model.eval()
            with torch.no_grad():
                _, embedding = self.detect(mels, class_labels.cuda(), self.model)

            self.embedding_list.append({'embedding': embedding,
                                        'domain_label': domain_labels,
                                        'class_label': class_labels})
        self.represent_embedding_dict = self.represent_extractor(pooling_type=self.hparams["represent"]["pooling_type"],
                                                                 domain_represent=True)

    def test_step(self, batch, batch_indx):
        audio, class_labels, anomaly_labels, domain_labels, filenames = batch

        mels = self.mel_spec(audio)
        _, embedding = self.detect(mels, class_labels, self.model)

        detected_embedding_dict = {
            'embedding': embedding,
            'domain_label': domain_labels,
            'class_label': class_labels
        }

        anomaly_scores = compute_batch_anomaly_score(detected_embedding_dict,
                                                     self.represent_embedding_dict,
                                                     self.hparams["represent"]["score_type"])

        batch_predict_df = batched_preds(anomaly_scores,
                                         class_labels,
                                         anomaly_labels,
                                         domain_labels,
                                         filenames,
                                         self.test_data_label_dict)
        self.test_buffer = self.test_buffer.append(batch_predict_df)

    def on_test_epoch_end(self):
        log_dir = self.logger.log_dir
        self.test_buffer.columns = ["filename", "anomaly_score", "anomaly_label",
                                    "machine_label", "section_label", "domain_label"]
        result_dict = compute_test_auc(self.test_buffer, max_fpr=self.hparams["training"]["max_fpr"])

        csv_line = []
        for machine, result_df in result_dict.items():
            csv_line.append([machine])
            csv_line.append(list(result_df.columns))
            for _, row in result_df.iterrows():
                csv_line.append(list(row))
            csv_line.append([])
        with open(os.path.join(log_dir, 'result.csv'), "w", newline="") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(csv_line)

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams["training"]["batch_size"],
            shuffle=True,
            num_workers=self.num_workers
        )
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            shuffle=False,
            num_workers=self.num_workers
        )
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False
        )
        return self.test_loader
