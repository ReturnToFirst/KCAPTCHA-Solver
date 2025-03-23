from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from server import utils
import glob
import argparse
import os
import cv2

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

# Load the YOLO8 model
model = YOLO("best.pt")

# Export the model to ONNX format
model.export(format="onnx")

quant_pre_process("best.onnx", "best_pre.onnx")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="./calibration_dataset", help="calibration data set"
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


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = YOLODataReader(
        calibration_dataset_path, input_model_path
    )

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")


if __name__ == "__main__":
    main()