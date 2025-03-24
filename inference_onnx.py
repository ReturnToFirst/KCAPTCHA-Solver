# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# modified from:
# https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import utils

classes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="best.onnx", help='Checkpoint path to use')
    parser.add_argument("--input", type=str, help="KCAPTCHA image file to read")
    parser.add_argument("--label", type=str, default="labels.yaml", help="classID-to-Label mapped yaml path")
    parser.add_argument("--c_thres", type=float, default=0.5, help="Confidence threshold for filtering detections.", dest='c_thres')
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.", dest='iou_thres')

    args = parser.parse_args()

    onnx_model = args.model
    img_path = args.input
    session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    model_inputs = session.get_inputs()
    model_dtype = utils.onnx_type_to_np_dtype(model_inputs[0].type)
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]
    input_height = input_shape[3]

    labels = utils.load_labels(args.label)

    img = cv2.imread(img_path)
    img_data, pad = utils.preprocess(img, [input_width, input_height], model_dtype)

    outputs = session.run(None, {model_inputs[0].name: img_data})

    print("".join(map(lambda x: utils.cls_id_to_label(labels, x), utils.postprocess(outputs, pad, args.c_thres, args.iou_thres))))