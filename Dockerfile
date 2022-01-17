FROM  tiangolo/uvicorn-gunicorn-machine-learning
COPY ./src /app
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt