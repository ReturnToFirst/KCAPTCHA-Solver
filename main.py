from fastapi import FastAPI, File, UploadFile
import uvicorn
import argparse
import onnxruntime as ort
import cv2
import numpy as np
import utils

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="best.onnx", help='Checkpoint path to use', dest='model')
parser.add_argument("--c_thres", type=float, default=0.5, help="Confidence threshold for filtering detections.", dest='c_thres')
parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.", dest='iou_thres')
args = parser.parse_args()

onnx_model = args.model
session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

model_inputs = session.get_inputs()
model_dtype = utils.onnx_type_to_np_dtype(model_inputs[0].type)
input_shape = model_inputs[0].shape
input_width = input_shape[2]
input_height = input_shape[3]

app = FastAPI()

@app.post("/")
async def solve_kcaptcha(file: UploadFile):
    global input_shape, input_width, input_height

    bytes_image = await file.read()
    encoded_img = np.frombuffer(bytes_image, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    img_data, pad = utils.preprocess(img, [input_width, input_height], model_dtype)

    outputs = session.run(None, {model_inputs[0].name: img_data})

    return {"solve": "".join(map(lambda x: str(x), utils.postprocess(outputs, pad, args.c_thres, args.iou_thres)))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")