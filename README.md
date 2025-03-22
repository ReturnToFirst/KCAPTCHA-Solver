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
Pretrained checkpoint available in this [huggingface](https://huggingface.co/UselessNerd/YOLO-V8-S-KCAPTCHA).
1. Inference KCAPTCHA with below command
    ```
    python3 inference.py --model <checkpoint path> \
                         --input <CAPTCHA image path>
    ```
