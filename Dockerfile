FROM jupyter/scipy-notebook

RUN pip install joblib

RUN mkdir model
ENV MODEL_DIR=./model
ENV MODEL_FILE=model_rf.joblib
ENV METADATA_FILE=metadata.json

COPY train.py ./train.py
COPY inference.py ./inference.py
COPY data/ml_eng_ay_data.csv.gz ./ml_eng_ay_data.csv.gz

RUN python3 -W ignore train.py