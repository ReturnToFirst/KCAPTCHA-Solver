FROM python:3.13-slim

ENV MODEL_FILE_NAME=best_int8.onnx
RUN apt-get update && apt-get install wget -y
RUN mkdir -p /server
RUN mkdir -p /model
ADD . /server
WORKDIR /server
RUN pip3 install -r requirements-server.txt

EXPOSE 8000
ENTRYPOINT ["sh", "-c"]
CMD ["python main.py --model /model/${MODEL_FILE_NAME}"]