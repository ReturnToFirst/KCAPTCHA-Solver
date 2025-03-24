# KCAPTCHA-Solver
Simple [KCAPTCHA](http://www.captcha.ru/en/kcaptcha/) solver using [YOLOv8](https://github.com/ultralytics/ultralytics)

## Requirements

- [ultralytics](https://pypi.org/project/ultralytics/)

## Usage

### Training

1. Install requirements with below command
    ```
    pip3 install ultralytics
    ```

2. Generate KCAPTCHA dataset
    1. Generate Image-label set with [kcaptcha-generator](https://github.com/ryanking13/kcaptcha-generator)
        - Recommand to generate >50K image-label set for training
    2. Split them into train, test, val set
    3. Reformat BBOX label to YOLO label format using `bbox2yolo.py`
        ```
        python3 bbox2yolo.py --path <dataset root path>
        ```

2. Configure your training config files as you want

2. Train YOLO with below command
    ```
    python3 train.py
    ```

    There is also options for training. You can get option by this command.

    ```
    python3 train.py --help
    ```

### Inference
Pretrained checkpoint available in [huggingface](https://huggingface.co/UselessNerd/YOLO-V8-S-KCAPTCHA).
1. Inference KCAPTCHA with below command
    ```
    python3 inference.py --model <checkpoint path> \
                         --input <CAPTCHA image path>
    ```
On test, Model's benchmark accuracy is 97.6%

### Onnx export
1. Export model to onnx with below command
    ```
    python3 export_to_onnx.py --input <model path>
    ```
#### Quantizaion
1. Prepare kcaptcha dataset(for best, prepare different dataset that doesn't used on training)
2. Export Quantized model with below command
    ```python3
    python3 export_to_onnx.py \
        --input <input yolo model path> \
        --output <output onnx model path> \
        --quantize \
        --calibration-dataset <calibration dataset path>
    ```

### Server
1. Start inference server with below command
    ```bash
    python3 main.py \
        --model <onnx model path> \
        --host 0.0.0.0 \
        --port 8000 \
        --c_thres 0.5 \
        --iou_thres 0.5
    ```

default exposed port is `8000`

### Docker

#### Docker run
```bash
docker run -d --restart always \
  -e MODEL_FILE_NAME=best_int8.onnx \
  -p 8000:8000 \
  -v ~/.model:/model \
  ghcr.io/returntofirst/kcaptcha-solver:latest
```

deploy model file on ~/.model and run this command.

#### Docker-compose
Use [docker-compose.yaml](docker-compose.yaml) to deploy with docker-compose.

deploy model file on ./model and run this command.

#### Kubenetes
Use [k8s-example.yaml](k8s-example.yaml) to deploy with kubectl.