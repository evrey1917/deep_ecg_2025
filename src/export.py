from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

# Импорт класса модели из вашего модуля
from model import ECGModel


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def export_to_onnx(cfg: DictConfig):
    """
    Конвертирует обученный чекпоинт PyTorch Lightning в формат ONNX,
    используя параметры из конфигурации препроцессинга.
    """
    # 1. Поиск чекпоинта и путей (берем из конфига)
    checkpoint_path = cfg.train.get("checkpoint_path", "models/last.ckpt")
    onnx_path = cfg.train.get("model_path", "models/ecg_model.onnx")

    # Извлекаем window_size из конфига препроцессинга (default.yaml)
    # Если в конфиге нет такого поля, используем 187 как значение по умолчанию
    window_size = cfg.preprocess.get("window_size", 187)

    if not Path(checkpoint_path).exists():
        print(f"Ошибка: Чекпоинт {checkpoint_path} не найден. Сначала обучите модель.")
        return

    print(f"Загрузка модели из чекпоинта: {checkpoint_path}")
    print(f"Размер входного окна (window_size): {window_size}")

    # 2. Инициализация модели и загрузка весов
    # Используем метод load_from_checkpoint из PyTorch Lightning
    try:
        model = ECGModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
        model.eval()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    # 3. Создание примера входных данных (Dummy Input)
    # Размерность: (batch_size, channels, sequence_length)
    dummy_input = torch.randn(1, 1, window_size)

    # 4. Экспорт
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Экспорт модели в формат ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,  # сохранять обученные веса внутри файла
        opset_version=18,  # версия ONNX
        do_constant_folding=True,  # оптимизация графа
        input_names=["input"],  # имя входного узла
        output_names=["output"],  # имя выходного узла
        dynamic_axes={  # поддержка переменного размера батча при инференсе
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("Экспорт успешно завершен. Модель готова к использованию в src/infer.py.")


if __name__ == "__main__":
    export_to_onnx()
