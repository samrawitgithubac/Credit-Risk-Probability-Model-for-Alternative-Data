FROM python:3.11-slim

WORKDIR /app

# System deps (optional but helps with some pip wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

EXPOSE 8000

# Env defaults (can be overridden by docker-compose)
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_URI=models:/credit_risk_model/1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


