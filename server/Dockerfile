FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

COPY server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]
