# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# modified from:
# https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py

import argparse
import cv2
import numpy as np
import onnxruntime as ort

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

def preprocess(input_img, new_shape):
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img, pad = letterbox(img, new_shape)

    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

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

            left = x - w
            top = y - h / 2
            width = w
            height = h

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    indices.sort()

    for i in indices:
        detected_classes.append(int(class_ids[i]))
    return detected_classes


classes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="best.onnx", help='Checkpoint path to use', dest='model')
    parser.add_argument("--input", type=str, help="KCAPTCHA image file to read", dest='input')
    parser.add_argument("--c_thres", type=float, default=0.5, help="Confidence threshold for filtering detections.", dest='c_thres')
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.", dest='iou_thres')

    args = parser.parse_args()

    onnx_model = args.model
    img_path = args.input
    session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]
    input_height = input_shape[3]

    img = cv2.imread(img_path)
    img_data, pad = preprocess(img, [input_width, input_height])

    outputs = session.run(None, {model_inputs[0].name: img_data})

    print("".join(map(lambda x: str(x), postprocess(outputs, pad, args.c_thres, args.iou_thres))))