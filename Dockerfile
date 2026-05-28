# Используем легкую версию Python 3.11
FROM python:3.11-slim

# Устанавливаем системные библиотеки (нужны для OpenCV/PyTorch/Pillow)
# libgl1 - замена устаревшему libgl1-mesa-glx в Debian Trixie
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Устанавливаем рабочую папку
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта и модель
# .dockerignore исключит папку data/, но скопирует runs/ с моделью
COPY . .

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]