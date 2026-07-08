FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
CMD sh -c "python3 -c 'import data_loader, graph_builder, pattern_detector; print(\"preloading\")' && uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"