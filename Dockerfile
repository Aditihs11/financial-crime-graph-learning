# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose backend (8000) and frontend (8501)
EXPOSE 8000 8501

ENV BACKEND_HOST="0.0.0.0"
ENV BACKEND_PORT=8000

# Default command launches backend and frontend in parallel
CMD ["sh", "-c", "python backend/api.py & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
