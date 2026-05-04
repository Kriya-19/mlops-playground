# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Prevents .pyc files, enables log flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model server
COPY serve.py .

# model.pkl will be bind-mounted or copied in at deploy time
# ENV values injected by docker run command

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]