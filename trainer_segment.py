import os
import csv
import torch
import random

import pandas as pd
import pytorch_lightning as pl

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Spectrogram
from torchmetrics.classification import Accuracy, AUROC
from tqdm import tqdm
from utils.scaler import TorchScaler
from utils.metric import batched_preds, compute_segment_test_auc, compute_batch_anomaly_score, represent_extractor, \
    compute_segment_batch_anomaly_score
from utils.data_aug import mixup, add_noise


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
            represent_loader=None,
            center_loss=None,
            opt_center_loss=None
    ):
        super(ASDTask, self).__init__()

        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)

        try:
            self.log_dir = self.logger.log_dir
        except Exception as e:
            self.log_dir = os.path.join(self.hparams["log_dir"], self.hparams["version"])
            os.makedirs(self.log_dir, exist_ok=True)

        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.represent_loader = represent_loader
        self.test_data_label_dict = {v: k for k, v in test_data.label_dict.items()}
        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        if self.hparams["feats"]["feature"] == "STFT":
            self.feat_transform = Spectrogram(
                n_fft=feat_params["n_window"] - 1,
                win_length=feat_params["n_window"] - 1,
                hop_length=feat_params["hop_length"],
                window_fn=torch.hamming_window,
                wkwargs={"periodic": False}
            )
        elif self.hparams["feats"]["feature"] == "Mel":
            self.feat_transform = MelSpectrogram(
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
        else:
            raise NotImplementedError

        self.scaler = self._init_scaler()

        self.supervised_loss = torch.nn.CrossEntropyLoss()
        self.center_loss = center_loss
        self.opt_center_loss = opt_center_loss

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
            transform_func=lambda x: self.take_log(self.feat_transform(x[0])),
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

    def training_step(self, batch, batch_idx):
        if self.hparams["represent"]["domain_represent"]:
            audio, class_labels, domain_labels = batch
        else:
            audio, class_labels = batch

        features = self.feat_transform(audio)

        # mixup
        if self.hparams["training"]["mixup"] and 0.5 > random.random():
            features, class_labels = mixup(features, class_labels)
        # add noise
        if self.hparams["training"]["add_noise"]:
            features = add_noise(features)

        preds, embeddings = self.detect(features, class_labels, self.model)

        loss = self.supervised_loss(preds, class_labels)
        if self.center_loss is not None:
            loss_cent = self.center_loss(embeddings, class_labels)
            loss += loss_cent

        self.log('train/step', self.global_step)
        self.log('train/loss', loss)
        self.log('train/lr', self.opt.param_groups[-1]["lr"])

        return loss

    def on_validation_epoch_start(self):
        self.embedding_list = []
        for batch in tqdm(self.represent_loader):

            if self.hparams["represent"]["domain_represent"]:
                audio, class_labels, domain_labels = batch
            else:
                audio, class_labels = batch
                domain_labels = None

            mels = self.feat_transform(audio.cuda())

            self.model.eval()
            with torch.no_grad():
                _, embedding = self.detect(mels, class_labels.cuda(), self.model)

            self.embedding_list.append({'embedding': embedding,
                                        'domain_label': domain_labels,
                                        'class_label': class_labels})
        self.represent_embedding_dict = represent_extractor(embedding_list=self.embedding_list,
                                                            pooling_type=self.hparams["represent"]["pooling_type"],
                                                            domain_represent=self.hparams["represent"]["domain_represent"])

    def validation_step(self, batch, batch_indx):
        if self.hparams["represent"]["domain_represent"]:
            audio, class_labels, anomaly_labels, domain_labels = batch
        else:
            audio, class_labels, anomaly_labels = batch
            domain_labels = None

        mels = self.feat_transform(audio)
        preds, embedding = self.detect(mels, class_labels, self.model)

        detected_embedding_dict = {
            'embedding': embedding,
            'domain_label': domain_labels,
            'class_label': class_labels
        }

        anomaly_score = compute_segment_batch_anomaly_score(detected_embedding_dict,
                                                            self.represent_embedding_dict,
                                                            self.hparams["represent"]["score_type"],
                                                            self.hparams["represent"]["domain_represent"]).cuda()

        batch_size = class_labels.shape[0]
        anomaly_labels = torch.stack([i[0] for i in torch.chunk(anomaly_labels, batch_size // 10, dim=0)], dim=0)
        class_labels = class_labels.to(torch.int16)

        self.accuracy_calculator.update(preds, class_labels)
        self.auc_calculator.update(anomaly_score, anomaly_labels)
        self.pauc_calculator.update(anomaly_score, anomaly_labels)

        return

    def validation_epoch_end(self, outputs):

        accuracy = self.accuracy_calculator.compute()
        auc = self.auc_calculator.compute()
        pauc = self.pauc_calculator.compute()

        self.log('valid/accuracy', accuracy)
        self.log('valid/auc+pauc', auc + pauc)
        self.log('valid/auc', auc, prog_bar=True)
        self.log('valid/pauc', pauc)
        return auc

    def on_save_checkpoint(self, checkpoint):
        checkpoint = self.model.state_dict()
        return checkpoint

    def on_test_epoch_start(self):
        self.embedding_list = []
        self.train_data.return_domain_label = True
        for batch in self.represent_loader:
            audio, class_labels, domain_labels = batch
            mels = self.feat_transform(audio.cuda())
            self.model.eval()
            with torch.no_grad():
                _, embedding = self.detect(mels, class_labels.cuda(), self.model)

            self.embedding_list.append({'embedding': embedding,
                                        'domain_label': domain_labels,
                                        'class_label': class_labels})
        self.represent_embedding_dict = represent_extractor(embedding_list=self.embedding_list,
                                                            pooling_type=self.hparams["represent"]["pooling_type"],
                                                            domain_represent=self.hparams["represent"]["domain_represent"])

    def test_step(self, batch, batch_indx):
        audio, class_labels, anomaly_labels, domain_labels, filenames = batch

        mels = self.feat_transform(audio)
        _, embedding = self.detect(mels, class_labels, self.model)

        detected_embedding_dict = {
            'embedding': embedding,
            'domain_label': domain_labels,
            'class_label': class_labels
        }
        batch_size = anomaly_labels.shape[0]
        anomaly_scores = compute_segment_batch_anomaly_score(detected_embedding_dict,
                                                             self.represent_embedding_dict,
                                                             self.hparams["represent"]["score_type"],
                                                             self.hparams["represent"]["domain_represent"]).cuda()

        anomaly_labels = torch.stack([i[0] for i in torch.chunk(anomaly_labels, batch_size // 10, dim=0)], dim=0)
        class_labels = torch.stack([i[0] for i in torch.chunk(class_labels, batch_size // 10, dim=0)], dim=0)
        domain_labels = tuple(list(domain_labels)[i:i+10][0] for i in range(batch_size // 10))
        filenames = tuple(list(filenames)[i:i+10][0] for i in range(batch_size // 10))

        batch_predict_df = batched_preds(anomaly_scores,
                                         class_labels,
                                         anomaly_labels,
                                         domain_labels,
                                         filenames,
                                         self.test_data_label_dict)
        self.test_buffer = self.test_buffer.append(batch_predict_df)

    def on_test_epoch_end(self):
        self.test_buffer.columns = ["filename", "anomaly_score", "anomaly_label",
                                    "machine_label", "section_label", "domain_label"]
        result_dict = compute_segment_test_auc(self.test_buffer, max_fpr=self.hparams["training"]["max_fpr"])

        csv_line = []
        auc_pauc = []
        for machine, result_df in result_dict.items():
            csv_line.append([machine])
            csv_line.append(list(result_df.columns))
            for _, row in result_df.iterrows():
                csv_line.append(list(row.apply(lambda x: format(x, '.2%') if not isinstance(x, str) else x)))
            auc_pauc.append('{:.2f}%/{:.2f}%'.format(result_df.loc[result_df['section'] == 'mean', 'auc'].values[0] * 100,
                                                     result_df.loc[result_df['section'] == 'mean', 'pauc'].values[0] * 100))
            csv_line.append([])

        csv_line.append(result_dict.keys())
        csv_line.append(['AUC/pAUC'] * len(result_dict.keys()))
        csv_line.append(auc_pauc)

        csv_save_path = os.path.join(self.log_dir, 'result.csv')
        with open(csv_save_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(csv_line)
        print(f"The results of the experiment are saved in {self.log_dir}")

    def configure_optimizers(self):
        if self.opt_center_loss is not None:
            return [self.opt, self.opt_center_loss], [self.scheduler]
        else:
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
