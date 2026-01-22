from pathlib import Path

import hydra
import numpy as np
import onnxruntime as ort
import wfdb
from omegaconf import DictConfig


def process_wfdb_file(record_path: str, window_size: int):
    """
    Загружает данные в формате WFDB (.hea/.dat) и готовит сегмент для модели.
    """
    # Загружаем запись (автоматически ищет .hea и .dat)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]  # Берем первое отведение (Lead I/II)

    # Минимальная нормализация (Min-Max), идентичная этапу обучения
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

    if len(signal) < window_size:
        pad_width = window_size - len(signal)
        signal = np.pad(signal, (0, pad_width), "constant")

    segment = signal[:window_size]
    return segment.astype(np.float32)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Путь к записи берется из конфига (например, data/raw/100)
    # По умолчанию используем тестовый файл из сырых данных
    input_path = cfg.get("input_path", "data/mit_bih/100")
    model_path = cfg.get("model_path", "models/ecg_model.onnx")
    window_size = cfg.preprocess.get("window_size", 187)

    path = Path(input_path)

    record_base = str(path.with_suffix(""))

    if not (Path(f"{record_base}.hea").exists() or Path(f"{record_base}.dat").exists()):
        print(f"Ошибка: Файлы WFDB записи {record_base} не найдены.")
        return

    # 1. Загрузка и препроцессинг данных из сырых файлов
    print(f"Загрузка и обработка WFDB записи: {record_base}")
    data = process_wfdb_file(record_base, window_size)

    # Добавляем размерности до (batch_size, channels, length) -> (1, 1, 187)
    data = data[np.newaxis, np.newaxis, :]

    # 2. Получение карты классов из конфига Hydra (секция preprocess)
    if "preprocess" in cfg and "label_map" in cfg.preprocess:
        label_map = cfg.preprocess.label_map
    else:
        # Фолбэк для стандартных классов AAMI
        label_map = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}

    inv_label_map = {v: k for k, v in label_map.items()}

    # 3. Запуск ONNX сессии
    if not Path(model_path).exists():
        print(f"Ошибка: Файл модели {model_path} не найден.")
        return

    session = ort.InferenceSession(model_path)
    inputs = {session.get_inputs()[0].name: data}
    logits = session.run(None, inputs)[0]

    # 4. Обработка результата
    prediction_idx = int(np.argmax(logits, axis=1)[0])
    label = inv_label_map.get(prediction_idx, "Unknown")

    # Расчет вероятностей (Softmax) для оценки уверенности
    exp_logits = np.exp(logits - np.max(logits))
    probability = exp_logits / np.sum(exp_logits)

    print("\n--- Результат анализа (WFDB + Hydra) ---")
    print(f"Запись: {path.name}")
    print(f"Класс сокращения: {label}")
    print(f"Уверенность модели: {np.max(probability) * 100:.2f}%")


if __name__ == "__main__":
    main()
