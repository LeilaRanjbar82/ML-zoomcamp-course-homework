FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

Copy ["predict-q6.py", "./"]

EXPOSE 8888

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8888", "predict-q6:app"]