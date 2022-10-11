import argparse
import warnings

warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
import yaml

from utils.GPU import auto_gpu

auto_gpu()

import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import ASDDataset
from label import dev_section_label_dict, dev_eval_section_label_dict
from nnet.model import MobileFaceNet, SimpleMobileFaceNet
from nnet.arcface import ArcMarginProduct
from nnet.center_loss import CenterLoss
from trainer import ASDTask


def main(
        config,
        log_dir,
        gpus,
        fast_dev_run=False,
        test_state_dict=None,
):
    config.update({'log_dir': log_dir})

    if config["training"]["use_eval"]:
        dev_train_df = pd.read_csv(config["data"]["dev_train_csv"])
        eval_train_df = pd.read_csv(config["data"]["eval_train_csv"])
        train_df = pd.concat([dev_train_df, eval_train_df])
        train_dataset = ASDDataset(
            audio_folder=config["data"]["audio_folder"],
            csv_entries=train_df,
            class_label_dict=dev_eval_section_label_dict,
            pad_to=config["data"]["audio_max_len"],
            dir_name='train',
            return_class_label=True,
            return_domain_label=config["represent"]["domain_represent"]
        )
        num_class = len(dev_eval_section_label_dict.keys())
    else:
        train_df = pd.read_csv(config["data"]["dev_train_csv"])
        train_dataset = ASDDataset(
            audio_folder=config["data"]["audio_folder"],
            csv_entries=train_df,
            class_label_dict=dev_section_label_dict,
            pad_to=config["data"]["audio_max_len"],
            dir_name='train',
            return_class_label=True,
            return_domain_label=config["represent"]["domain_represent"]
        )
        num_class = len(dev_section_label_dict)

    test_df = pd.read_csv(config["data"]["dev_test_csv"])
    test_dataset = ASDDataset(
        audio_folder=config["data"]["audio_folder"],
        csv_entries=test_df,
        class_label_dict=dev_section_label_dict,
        pad_to=config["data"]["audio_max_len"],
        dir_name='test',
        return_class_label=True,
        return_anomaly_label=True,
        return_domain_label=True,
        return_filename=True
    )

    ########################
    # define arcface
    ########################
    if config["training"]["arcface"]:
        arcface = ArcMarginProduct(config["net"]["embedding_size"],
                                   config["net"]["num_class"],
                                   m=config["net"]["margin"],
                                   s=config["net"]["scale"])
    else:
        arcface = None

    ########################
    # define model
    ########################
    model = MobileFaceNet(num_class=config["net"]["num_class"],
                          arcface=arcface,
                          embedding_size=config["net"]["embedding_size"])

    ########################
    # define center loss
    ########################
    if config["training"]["center_loss"]:
        criterion_cent = CenterLoss(num_classes=config["net"]["num_class"],
                                    feat_dim=config["net"]["embedding_size"])
        opt_center_loss = torch.optim.Adam(criterion_cent.parameters(), config["opt"]["center_loss_lr"])
    else:
        criterion_cent = None
        opt_center_loss = None


    if test_state_dict is None:
        valid_df = pd.read_csv(config["data"]["dev_test_csv"])
        valid_dataset = ASDDataset(
            audio_folder=config["data"]["audio_folder"],
            csv_entries=valid_df,
            class_label_dict=dev_section_label_dict,
            pad_to=config["data"]["audio_max_len"],
            dir_name='test',
            return_class_label=True,
            return_anomaly_label=True,
            return_domain_label=config["represent"]["domain_represent"]
        )

        opt = torch.optim.Adam(model.parameters(), config["opt"]["lr"], betas=(0.9, 0.999))
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99),
            "interval": "epoch"
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
            version=config["version"]
        )

        callbacks = [
            EarlyStopping(
                monitor="valid/auc",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="valid/auc",
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
        ]
    else:
        valid_dataset = None
        batch_sampler = None
        opt = None
        scheduler = None
        logger = False
        callbacks = None

    ASD_training = ASDTask(
        config,
        model=model,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        scheduler=scheduler,
        fast_dev_run=fast_dev_run,
        center_loss=criterion_cent,
        opt_center_loss=opt_center_loss
    )

    if fast_dev_run:
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 1.0
        n_epochs = 1
    else:
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=gpus,
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )
    if test_state_dict is None:
        trainer.fit(ASD_training)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best_model_path: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]
    ASD_training.load_state_dict(test_state_dict)
    trainer.test(ASD_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a ASD system")
    parser.add_argument(
        "--conf_file",
        default="./confs/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--version",
        default="test",
        help="Record your experimental details to distinguish others experiment",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
             "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
             "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    test_from_checkpoint = args.test_from_checkpoint
    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        print(f"loaded model from: {test_from_checkpoint}")
        test_model_state_dict = checkpoint["state_dict"]

    seed = configs["training"]["seed"]
    configs.update({'version': args.version})
    if seed:
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.backends.cudnn.deterministic = True

    main(
        configs,
        args.log_dir,
        args.gpus,
        args.fast_dev_run,
        test_model_state_dict
    )
