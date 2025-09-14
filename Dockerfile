FROM python:3.11-slim

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY models/production ./models/production

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8080"]
