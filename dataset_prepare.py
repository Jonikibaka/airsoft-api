"""
Скрипт подготовки датасета для классификации изображений.
Исправляет опечатки в названиях классов и делит данные на train/val/test.
"""

import os
import shutil
import random
from pathlib import Path

# === НАСТРОЙКИ ===
SOURCE_DIR = Path("./data/raw/dataset")
OUTPUT_DIR = Path("./data/processed/dataset_cls")

# Пропорции разделения
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Тест займёт оставшиеся 10%

# Словарь для исправления опечаток и приведения к единому формату
CLASS_NAME_MAP = {
    "M serias": "m_series",
    "mashinegun": "machinegun",
    "shutgun": "shotgun",
    "HK": "hk",
    # Остальные папки уже написаны корректно, их просто переведём в нижний регистр
}

def prepare_dataset():
    """Основная логика подготовки датасета"""
    print("Начинаю подготовку датасета для классификации...")

    # Если папка уже существует, очищаем её для чистоты эксперимента
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f" Удалена старая папка {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Создаём корневые папки для сплитов
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir()

    # Поддерживаемые расширения изображений
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    total_images_moved = 0

    # Проходим по каждой папке с исходными данными
    for class_folder in SOURCE_DIR.iterdir():
        if not class_folder.is_dir():
            continue

        raw_name = class_folder.name
        
        # Применяем маппинг или просто переводим в нижний регистр
        clean_name = CLASS_NAME_MAP.get(raw_name, raw_name.lower())
        
        print(f"Обработка: '{raw_name}' → '{clean_name}'")

        # Собираем все пути к изображениям в папке
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in supported_exts]
        
        # Перемешиваем для случайного разделения
        random.shuffle(images)
        
        total_count = len(images)
        if total_count == 0:
            print(f"Нет изображений, пропускаю.")
            continue

        # Вычисляем границы разрезов
        train_end = int(total_count * TRAIN_RATIO)
        val_end = int(total_count * VAL_RATIO) + train_end

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        # Копируем файлы в соответствующие папки
        for split_name, split_images in splits.items():
            target_dir = OUTPUT_DIR / split_name / clean_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest_path = target_dir / img_path.name
                shutil.copy2(img_path, dest_path)  # copy2 сохраняет метаданные файла

            total_images_moved += len(split_images)
        
        print(f"Готово: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print(f" Структура сохранена в: {OUTPUT_DIR}")
    print(f" Всего скопировано изображений: {total_images_moved}")

if __name__ == "__main__":
    # Фиксируем seed, чтобы разделение было воспроизводимым при повторных запусках
    random.seed(42)
    prepare_dataset()