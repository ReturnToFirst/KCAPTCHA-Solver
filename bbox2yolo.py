import glob
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='./dataset', help='dataset root path',dest='path')

args = parser.parse_args()
label_file_paths = glob.glob(os.path.join(args.path, '*/*.json'))

def parse_bbox_label(json_path):
    with open(json_path, 'r') as json_read:
        label_dict = json.load(json_read)
    image_height = label_dict["height"]
    image_width = label_dict["width"]
    bboxs = label_dict["bbox"]
    return image_height, image_width, bboxs

def make_yolo_label(image_height, image_width, bbox):
    label = bbox["label"]
    x_min = bbox["xmin"]
    y_min = bbox["ymin"]
    x_max = bbox["xmax"]
    y_max = bbox["ymax"]
    x_center = get_center(image_width, x_min, x_max)
    y_center = get_center(image_height, y_min, y_max)
    label_width = (x_max - x_min) / image_width
    label_height = (y_max - y_min) / image_height

    return f'{label} {x_center} {y_center} {label_width} {label_height}'

def get_center(length, min, max):
    abs_center = (min + max) / 2
    rel_center = abs_center / length
    return rel_center

for label_file_path in label_file_paths:
    pwd_path, file_name_with_ext = os.path.split(label_file_path)
    _, folder_path = os.path.split(pwd_path)
    file_name = os.path.splitext(file_name_with_ext)[0]

    img_h, img_w, bboxs = parse_bbox_label(label_file_path)
    yolo_label = list()
    for bbox in bboxs:
        yolo_label.append(make_yolo_label(img_h, img_w, bbox))

    with open(os.path.join(args.path, folder_path, f'{file_name}.txt'), 'w') as label_writer:
         label_writer.write("\n".join(yolo_label))
    os.remove(label_file_path)
        