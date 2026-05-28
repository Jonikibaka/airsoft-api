"""
Скрипт для обучения модели классификации YOLOv8.
Использует подготовленные данные из data/processed/dataset_cls
"""

from ultralytics import YOLO
import os
from pathlib import Path

# === НАСТРОЙКИ ===
# Путь к датасету (YOLO автоматически ищет папки train/val/test внутри)
DATA_PATH = Path("./data/processed/dataset_cls")

# Параметры обучения
EPOCHS = 50          # Количество эпох (можно увеличить до 100 для лучшего результата)
IMAGE_SIZE = 640     # Размер изображения для обучения
BATCH_SIZE = 16      # Размер батча (зависит от мощности GPU, для CPU можно поставить 8 или 4)
MODEL_NAME = "yolov8n-cls.pt" # 'n' - nano (самая быстрая), 's' - small, 'm' - medium

def train_model():
    """
    Запускает процесс обучения модели классификации.
    """
    print(" Начинаем обучение модели классификации...")
    print(f" Датасет: {DATA_PATH}")
    print(f" Модель: {MODEL_NAME}")
    print(f" Эпохи: {EPOCHS}, Размер: {IMAGE_SIZE}x{IMAGE_SIZE}")


    # 1. Загрузка предобученной модели (используем imagenet для трансферного обучения)
    # transfer=False, так как мы обучаем с нуля на своих данных (или используем imagenet weights)
    model = YOLO(MODEL_NAME)

    # 2. Запуск тренировки
    # results содержит путь к папке с логами и графиками
    results = model.train(
        data=str(DATA_PATH), # Указываем корневую папку датасета
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name="airsoft_cls_v1", # Имя эксперимента (папка runs/cls/airsoft_cls_v1)
        project="runs",        # Базовая папка для результатов
        workers=4,             # Количество потоков для загрузки данных
        verbose=True           # Подробный вывод в консоль
    )

    print("Обучение завершено!")
    print(f"Метрики сохранены в: {results.save_dir}")
    print(f"Лучшая модель: {results.best}") # Обычно это weights/best.pt
    
    # Выводим итоговую точность (Top-1 accuracy)
    if hasattr(results, 'results_dict'):
        acc = results.results_dict.get('metrics/accuracy_top1', 0)
        print(f"Итоговая точность (Top-1): {acc * 100:.2f}%")

if __name__ == "__main__":
    train_model()