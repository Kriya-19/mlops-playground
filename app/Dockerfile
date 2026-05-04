FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.41.1 \
    requests==2.32.3 \
    pandas==2.2.3 \
    plotly==5.24.1

COPY app/ ./app/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
