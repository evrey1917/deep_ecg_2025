from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from datamodule import ECGDataModule
from model import ECGModel


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # 1. Проверка данных
    # Используем интерполяцию из конфига: cfg.data.processed_path
    # теперь ссылается на cfg.preprocess.processed_data_path
    processed_data_path = Path(cfg.data.processed_path)

    if not processed_data_path.exists():
        print(f"ERROR: Processed data not found at {processed_data_path}")
        print("Please run 'uv run python src/preprocess.py' first.")
        return

    # 2. Настройка MLflow логгера
    # Lightning автоматически запишет гиперпараметры и создаст графики
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    )

    # 3. Инициализация данных и модели
    dm = ECGDataModule(
        processed_path=str(processed_data_path),
        batch_size=cfg.train.batch_size,
        val_split=cfg.data.val_split,
    )

    model = ECGModel(cfg)

    # 4. Настройка Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        logger=mlf_logger,
        accelerator=cfg.train.accelerator,
        log_every_n_steps=10,  # Чтобы графики были плавнее
    )

    # 5. Обучение
    trainer.fit(model, dm)

    # 6. Сохранение артефактов
    models_dir = Path(cfg.paths.models_dir)
    models_dir.mkdir(exist_ok=True)

    checkpoint_path = Path(cfg.checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
