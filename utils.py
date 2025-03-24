import cv2
import numpy as np
import json
import yaml

def load_labels(path):
    with open(path) as yaml_read:
        return yaml.load(yaml_read, Loader=yaml.FullLoader)

def cls_id_to_label(labels, cls_id):
    return labels.get(cls_id, "")

def letterbox(img, new_shape):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, (top, left)

def preprocess(input_img, new_shape, dtype):
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img, pad = letterbox(img, new_shape)

    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(dtype)

    return image_data, pad

def postprocess(output, pad, confidence_thres, iou_thres):
    boxes = []
    scores = []
    class_ids = []
    detected_classes = []

    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]

    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)

            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            left = x - w / 2
            top = y - h / 2

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    detections = [(boxes[i][0], int(class_ids[i])) for i in indices]
    detections.sort(key=lambda x: x[0])

    detected_classes = [cls_id for _, cls_id in detections]
    return detected_classes

def onnx_type_to_np_dtype(dtype):
    onnx_to_numpy_type = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(int16)": np.int16,
        "tensor(uint16)": np.uint16,
        "tensor(int32)": np.int32,
        "tensor(uint32)": np.uint32,
        "tensor(int64)": np.int64,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }
    return onnx_to_numpy_type.get(dtype, float)
