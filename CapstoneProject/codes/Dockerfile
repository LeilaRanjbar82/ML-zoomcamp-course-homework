FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

COPY yoga_08_0.862.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]