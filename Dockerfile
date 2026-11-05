FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy the model artifact (REQUIRED for /predict)
COPY artifacts/ ./artifacts/

# Render provides PORT; 10000 is a convenient default
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --timeout 120 app:app"]
