import subprocess

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import ECGModel


class ECGDataModule(L.LightningDataModule):
    def __init__(self, processed_path, batch_size):
        super().__init__()
        self.path = processed_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        data = torch.load(self.path)
        full_ds = TensorDataset(data["x"], data["y"])

        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        self.train_ds, self.val_ds = random_split(full_ds, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # 1. DVC Pull (Требование задания)
    print("Syncing data via DVC...")
    subprocess.run(["dvc", "pull"], check=True)

    # 2. Настройка MLflow
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    )

    # 3. Инициализация данных и модели
    dm = ECGDataModule(cfg.data.train_path, cfg.data.test_path, cfg.train.batch_size)
    model = ECGModel(cfg)

    # 4. Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        logger=mlf_logger,
        accelerator=cfg.train.accelerator,
    )

    # 5. Обучение
    trainer.fit(model, dm)

    # Сохраняем модель локально для экспорта
    trainer.save_checkpoint("models/last.ckpt")


if __name__ == "__main__":
    train()
