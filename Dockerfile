FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY models /app/models

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=models/latest.keras

EXPOSE 8000

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
