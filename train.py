from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()

# Environment options
parser.add_argument("--config",  type=str, default='./configs/config.yaml', help="path to yolo config file", dest="config")
parser.add_argument("--log-path", type=str, default='captcha_solver', help="Relative path to save model", dest='path')
parser.add_argument("--devices", type=str, default='0', help="Devices to use on training", dest='devices')
parser.add_argument("--dataloader-workers", type=int, default='8', help='Dataloader workers count', dest='workers')
parser.add_argument("--img-size", type=int, default='160', help="max width of image", dest='img_size')

# Model options
parser.add_argument("--model", type=str, default='yolov8s', help="model to use", dest='model')

# Training options
parser.add_argument("--epochs", type=int, default=300, help='Epochs to train', dest='epochs')
parser.add_argument("--batch-size", type=int, default=16, help="batch size to use on training")
parser.add_argument("--no-finetune", action='store_false', help="Train network from scratch", dest='no_finetune')
parser.add_argument("--optimizer", type=str, default='NAdam', help="Optimizer to use", dest='optimizer')
parser.add_argument("--val", action='store_true', help='Use validation on training', dest='val')

args = parser.parse_args()

model = YOLO(f"{args.model}.yaml")

if args.no_finetune == False:
    model = model.load(f'{args.model}.yaml')


model.train(data=args.config,
            optimizer=args.optimizer,
            imgsz=args.img_size,
            project=args.path,
            batch=args.batch_size,
            val=True if args.val else False,
            workers=args.workers,
            epochs=args.epochs,
            device=args.devices,
            resume=True if not args.no_finetune else False,
            exist_ok=True)
