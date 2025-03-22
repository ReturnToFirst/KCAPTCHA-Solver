from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default="best.pt", help='Checkpoint path to use', dest='model')
parser.add_argument("--input-file", type=str, help="KCAPTCHA image file to read", dest='input')

args = parser.parse_args()

model = YOLO(args.model)

results = model(args.input)
result = results[0]
pred_res = result.boxes.data

# prediction result
# x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence   class

sorted_pred = sorted(pred_res, key=lambda pred: pred[0])
pred_captcha = list()
for pred in sorted_pred:
    pred_captcha.append(str(int(pred[5]))) # class
    
print(f"Predicted CAPTCHA: {''.join(pred_captcha)}")