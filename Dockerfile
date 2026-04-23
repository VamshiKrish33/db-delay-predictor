FROM python:3.12-slim

WORKDIR /app

# System deps required by XGBoost on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    joblib \
    fastapi \
    "uvicorn[standard]" \
    pydantic

# Copy source and model artefacts
COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
