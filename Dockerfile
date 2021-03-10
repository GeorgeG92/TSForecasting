FROM tensorflow/tensorflow:2.3.0-gpu

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY ./config ./config

COPY ./data ./data

COPY ./src ./src
