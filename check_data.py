# step1_check_data.py
"""
Скрипт для проверки структуры датасета.
Подсчитывает количество изображений в каждой папке и проверяет, 
что файлы действительно являются картинками.
"""

import os
from PIL import Image
import glob

# Путь к папке с сырыми данными (относительно этого скрипта)
# Убедитесь, что скрипт лежит в корне проекта рядом с папкой data
DATASET_PATH = "./data/raw/dataset"

def check_dataset(path):
    """
    Проходит по папкам, считает картинки, проверяет их целостность.
    """
    print(f" Сканирование датасета: {path}...\n")

    if not os.path.exists(path):
        print(f"❌ Ошибка: Папка {path} не найдена!")
        return

    total_images = 0
    classes = {}
    
    # Получаем список всех подпапок (названий классов)
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

    for folder in folders:
        folder_path = os.path.join(path, folder)
        
        # Ищем все картинки (jpg, jpeg, png)
        images = glob.glob(os.path.join(folder_path, "*.jpg"))
        images += glob.glob(os.path.join(folder_path, "*.jpeg"))
        images += glob.glob(os.path.join(folder_path, "*.png"))
        
        valid_count = 0
        error_count = 0
        
        for img_path in images:
            try:
                # Пытаемся открыть картинку, чтобы убедиться, что она не битая
                with Image.open(img_path) as img:
                    img.verify() 
                    valid_count += 1
            except Exception:
                error_count += 1
                print(f" Битый файл: {img_path}")

        classes[folder] = valid_count
        total_images += valid_count
        
        print(f" Папка '{folder}': {valid_count} картинок ({error_count} ошибок)")

    print("-" * 40)
    print(f" ИТОГО классов: {len(classes)}")
    print(f" ИТОГО картинок: {total_images}")
    print("-" * 40)
    
    # Выводим предупреждения про опечатки, если нашли их
    if "M serias" in classes:
        print(" Найдена папка с опечаткой 'M serias' (возможно, имелось в виду 'm_series').")
    if "mashinegun" in classes:
        print(" Найдена папка с опечаткой 'mashinegun' (возможно, имелось в виду 'machinegun').")

if __name__ == "__main__":
    check_dataset(DATASET_PATH)