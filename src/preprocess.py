from pathlib import Path

import hydra
import numpy as np
import torch
import wfdb
from omegaconf import DictConfig


def get_records(data_dir: Path):
    """Список записей MIT-BIH (имена без расширений)."""
    # .glob ищет файлы, .stem забирает имя без расширения
    hea_files = data_dir.glob("*.hea")
    return sorted({f.stem for f in hea_files})


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def preprocess(cfg: DictConfig):

    data_dir = Path(cfg.preprocess.raw_data_dir)
    processed_path = Path(cfg.preprocess.processed_data_path)

    if not data_dir.exists():
        print(f"Error: Raw data directory {data_dir} does not exist!")
        return

    records = get_records(data_dir)
    all_beats = []
    all_labels = []

    print(f"Processing {len(records)} records from {data_dir}...")

    for record_name in records:
        # Path позволяет склеивать пути через оператор /
        record_full_path = data_dir / record_name

        # wfdb принимает строку, поэтому конвертируем Path в str
        record = wfdb.rdrecord(str(record_full_path))
        ann = wfdb.rdann(str(record_full_path), "atr")

        signal = record.p_signal[:, 0]

        for idx, symbol in zip(ann.sample, ann.symbol):
            if symbol in cfg.preprocess.label_map:
                half_win = cfg.preprocess.window_size // 2
                if half_win < idx < len(signal) - half_win:
                    beat = signal[idx - half_win : idx + half_win + 1]

                    # Нормализация
                    denom = np.max(beat) - np.min(beat) + 1e-8
                    beat = (beat - np.min(beat)) / denom

                    all_beats.append(beat)
                    all_labels.append(cfg.preprocess.label_map[symbol])

    # 3. Сохранение результатов
    X = torch.FloatTensor(np.array(all_beats)).unsqueeze(1)
    y = torch.LongTensor(np.array(all_labels))

    # Создаем родительскую папку, если её нет
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"x": X, "y": y}, processed_path)
    print(f"Saved {len(X)} samples to {processed_path}")


if __name__ == "__main__":
    preprocess()
