FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY web ./web

ENV USDV_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
