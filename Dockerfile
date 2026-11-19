FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app

# Index, database and uploaded docs live in /app/data. Mount a volume there to keep them.
VOLUME ["/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"

# One worker: the local Qdrant index, SQLite and the rate limiter live in this process.
# --proxy-headers so X-Forwarded-For works behind Render/Railway/nginx (also set TRUST_PROXY=true).
# Hosting platforms (Render, Railway, Cloud Run) pass the port in $PORT; locally it is 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]
