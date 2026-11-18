FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# The index is built on first start from data/docs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
