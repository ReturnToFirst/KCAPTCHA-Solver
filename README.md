# KCAPTCHA-Solver
Simple Numerical [KCAPTCHA](http://www.captcha.ru/en/kcaptcha/) solver using [YOLOv8](https://github.com/ultralytics/ultralytics)

## Usage

### Training

1. Install requirements with below command
    ```bash
    pip3 install -r requirements.txt
    ```

2. Generate KCAPTCHA dataset
    1. Generate Image-label set with [kcaptcha-generator](https://github.com/ryanking13/kcaptcha-generator)
        - Recommand to generate >50K image-label set for training
    2. Split them into train, test, val set
    3. Reformat BBOX label to YOLO label format using `bbox2yolo.py`
        ```bash
        usage: bbox2yolo.py [-h] [--path PATH]

        options:
        -h, --help   show this help message and exit
        --path PATH  dataset root path
        ```

3. Configure your [training config files](configs/config.yaml) as you want

4. Train YOLO with below command
    ```bash
    usage: train.py [-h] [--config CONFIG] [--log-path PATH] [--devices DEVICES] [--dataloader-workers WORKERS] [--img-size IMG_SIZE] [--model MODEL] [--epochs EPOCHS]
                    [--batch-size BATCH_SIZE] [--no-finetune] [--optimizer OPTIMIZER] [--val]

    options:
    -h, --help            show this help message and exit
    --config CONFIG       path to yolo config file
    --log-path PATH       Relative path to save model
    --devices DEVICES     Devices to use on training
    --dataloader-workers WORKERS
                            Dataloader workers count
    --img-size IMG_SIZE   max width of image
    --model MODEL         model to use
    --epochs EPOCHS       Epochs to train
    --batch-size BATCH_SIZE
                            batch size to use on training
    --no-finetune         Train network from scratch
    --optimizer OPTIMIZER
                            Optimizer to use
    --val                 Use validation on training
    ```

### Inference
Pretrained checkpoint available in [huggingface](https://huggingface.co/UselessNerd/YOLO-V8-S-KCAPTCHA).

Inference KCAPTCHA with below command
```bash
usage: inference.py [-h] [--model MODEL] [--input INPUT] [--label LABEL]

options:
-h, --help     show this help message and exit
--model MODEL  Checkpoint path to use
--input INPUT  KCAPTCHA image file to read
--label LABEL  classID-to-Label mapped yaml path
```

On test, Model's benchmark accuracy is 97.6%

### Onnx export
Export model to onnx with below command
```bash
python3 export_to_onnx.py --input <model path>
```

#### Quantizaion
1. Prepare kcaptcha dataset(for best, prepare different dataset that doesn't used on training)
2. Export Quantized model with below command
    ```bash
    usage: export_to_onnx.py [-h] --input INPUT [--output OUTPUT] [--quantize] [--calibration-dataset CALIBRATION_DATASET] [--quant_format {QOperator,QDQ}]
                            [--per_channel PER_CHANNEL]

    options:
    -h, --help            show this help message and exit
    --input INPUT         input yolov8 torch model
    --output OUTPUT       output onnx model path for quantized model
    --quantize            flag for quantize onnx model. Require --calibration-dataset and --quant-format
    --calibration-dataset CALIBRATION_DATASET calibration data set path
    --quant_format {QOperator,QDQ}
    --per_channel PER_CHANNEL
    ```

### Onnx Inference
Inference KCAPTCHA with below command with onnx weight

```bash
usage: inference_onnx.py [-h] [--model MODEL] [--input INPUT] [--label LABEL] [--c_thres C_THRES] [--iou_thres IOU_THRES]

options:
-h, --help            show this help message and exit
--model MODEL         Checkpoint path to use
--input INPUT         KCAPTCHA image file to read
--label LABEL         classID-to-Label mapped yaml path
--c_thres C_THRES     Confidence threshold for filtering detections.
--iou_thres IOU_THRES IoU threshold for non-maximum suppression.      
```


### Server Usage
Pretrained checkpoint available in [huggingface](https://huggingface.co/UselessNerd/YOLO-V8-S-KCAPTCHA).

Only `ONNX` format weight accepted by server for performance reason.
Quantized model(`best_int8.onnx`) is highly recommanded.

default exposed port is `8000`

### Local inference

1. Install requirements with below command
    ```bash
    pip3 install -r requirements-server.txt
    ```
2. Start inference server with below command.
    ```bash
    usage: main.py [-h] [--host HOST] [--port PORT] [--model MODEL] [--label LABEL] [--c_thres C_THRES] [--iou_thres IOU_THRES]

    options:
    -h, --help            show this help message and exit
    --host HOST           Server host
    --port PORT           Server port
    --model MODEL         Checkpoint path to use
    --label LABEL         classID-to-Label mapped yaml path
    --c_thres C_THRES     Confidence threshold for filtering detections.
    --iou_thres IOU_THRES IoU threshold for non-maximum suppression.
    ```

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