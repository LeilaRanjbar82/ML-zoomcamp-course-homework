FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

Copy ["predict.py", "model.bin", "./"]

EXPOSE 8889

ENTRYPOINT ["gunicorn", "--bind", "localhost:8889", "predict:app"]