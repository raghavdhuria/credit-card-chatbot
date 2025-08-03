FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update -y && apt-get install -y awscli ffmpeg libsm6 libxext6 unzip

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
