FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Build deps for some wheels (rapidfuzz/Levenshtein can compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 8080
# production server
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
