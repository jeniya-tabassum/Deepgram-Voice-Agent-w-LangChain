FROM python:3.12-slim

WORKDIR /app

# Unbuffered so the startup graph + [brain] step logs show up in `fly logs`.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install deps first (better layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + static frontend.
COPY app ./app
COPY frontend ./frontend

# Fly routes to this internal port (see fly.toml).
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
