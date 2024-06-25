from ultralytics import YOLO

MODEL = './models/coco128/yolov8-4detect.pt'
DATASET = 'coco128.yaml'
VAL = 'val'

if __name__ == '__main__':
    model = YOLO(MODEL)
    model.val(
        data=DATASET,
        split=VAL,
        imgsz=640,
        batch=32,
        project='runs/detect/test'
    )
