import os

import hydra
import numpy as np
import torch
import wfdb
from omegaconf import DictConfig


def get_records(data_dir):
    # Получаем список всех записей (имена файлов без расширений)
    files = [f.split(".")[0] for f in os.listdir(data_dir) if f.endswith(".hea")]
    return sorted(list(set(files)))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def preprocess(cfg: DictConfig):
    data_dir = cfg.preprocess.raw_data_dir
    records = get_records(data_dir)

    all_beats = []
    all_labels = []

    print(f"Processing {len(records)} records...")

    for record_name in records:
        record_path = os.path.join(data_dir, record_name)
        # Читаем сигнал и аннотации
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, "atr")

        signal = record.p_signal[:, 0]  # первый канал

        for idx, symbol in zip(ann.sample, ann.symbol):
            if symbol in cfg.preprocess.label_map:
                # Центрируем окно вокруг R-пика
                half_win = cfg.preprocess.window_size // 2
                if idx > half_win and idx < len(signal) - half_win:
                    beat = signal[idx - half_win : idx + half_win + 1]
                    # Нормализация (амплитуда 0-1)
                    beat = (beat - np.min(beat)) / (np.max(beat) - np.min(beat) + 1e-8)

                    all_beats.append(beat)
                    all_labels.append(cfg.preprocess.label_map[symbol])

    # Сохраняем как тензоры PyTorch (удобно для Lightning)
    X = torch.FloatTensor(np.array(all_beats)).unsqueeze(1)
    y = torch.LongTensor(np.array(all_labels))

    os.makedirs(os.path.dirname(cfg.preprocess.processed_data_path), exist_ok=True)
    torch.save({"x": X, "y": y}, cfg.preprocess.processed_data_path)
    print(f"Saved {len(X)} samples to {cfg.preprocess.processed_data_path}")


if __name__ == "__main__":
    preprocess()
