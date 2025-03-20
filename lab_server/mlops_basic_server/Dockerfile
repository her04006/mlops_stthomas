# This dockerfile is for our kubernetes cluster
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yeah they all get mashed together ;-)
COPY src/ .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]