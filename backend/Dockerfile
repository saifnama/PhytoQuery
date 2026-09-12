# PhytoQuery API image. Build from repo root:
#   docker build -f backend/Dockerfile -t phytoquery-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOKY_MAX_CPU_COUNT=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && python -m spacy download en_core_web_sm

COPY backend/ ./backend/
COPY data/db/ ./data/db/

EXPOSE 8000

CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
