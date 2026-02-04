FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаем директорию для данных и файл products.json если его нет
RUN mkdir -p /app/data && \
    touch /app/products.json || true

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Открываем порт
EXPOSE 8080

# Проверяем, что файлы на месте
RUN ls -la /app/

# Запускаем приложение
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
