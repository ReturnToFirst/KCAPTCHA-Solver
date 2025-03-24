FROM python:3.13-slim

ENV MODEL_FILE_NAME=best_int8.onnx
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV YOLO_LABEL_FILE=/labels.yaml
ENV YOLO_CONFIDENCE_THRESHOLD=0.5
ENV YOLO_IOU_THRESHOLD=0.5

RUN apt-get update && apt-get install wget -y
RUN mkdir -p /server
RUN mkdir -p /model

ADD . /server
WORKDIR /server
RUN pip3 install -r requirements-server.txt

EXPOSE 8000

ENTRYPOINT ["sh", "-c"]
CMD ["python main.py --model /model/${MODEL_FILE_NAME} --host ${SERVER_HOST} --port ${SERVER_PORT} --label=${YOLO_LABEL_FILE} --c_thres ${YOLO_CONFIDENCE_THRESHOLD} --iou_thres ${YOLO_IOU_THRESHOLD}"]