import onnxruntime as ort
import glob
import argparse
import os
import cv2
import utils
from ultralytics import YOLO
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static



class YOLODataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = ort.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape
        model_inputs = session.get_inputs()
        model_dtype = utils.onnx_type_to_np_dtype(model_inputs[0].type)

        # Convert image to input data
        self.data_list = []
        for path in glob.glob(os.path.join(calibration_image_folder, "*.png")):
            img = cv2.imread(path)
            self.data_list.append(utils.preprocess(img, (height, width), model_dtype)[0])
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: data} for data in self.data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input yolov8 torch model")
    parser.add_argument("--output", default="kcaptcha_yolo_int8.onnx", help="output onnx model path for quantized model", type=str)
    parser.add_argument("--quantize", action="store_true", help="flag for quantize onnx model. Require --calibration-dataset and --quant-format")
    parser.add_argument(
        "--calibration-dataset", default="./calibration_dataset", help="calibration data set path"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    # Load the YOLO8 model
    model = YOLO(args.input)

    # Export the model to ONNX format
    model.export(format="onnx")
    print("Onnx model saved.")
    if args.quantize:
        print("Start Calibration/Quanti")
        output_model_path = args.output

        file_name = os.path.splitext(os.path.split(args.input)[1])[0]
        preprocessed_model = f"{file_name}_infer.onnx"

        quant_pre_process(f"{file_name}.onnx", preprocessed_model)
        print("Quantization model preprocessing done.")

        dr = YOLODataReader(
            args.calibration_dataset, preprocessed_model
        )
        # Calibrate and quantize model
        # Turn off model optimization during quantization
        print("Start quantization")
        quantize_static(
            preprocessed_model,
            output_model_path,
            dr,
            quant_format=args.quant_format,
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8,
        )
        print("Calibrated and quantized model saved.")