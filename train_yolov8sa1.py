from ultralytics import YOLO

DATASET = r'coco128.yaml'

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-sa1.yaml')
    model = model.load('yolov8n.pt')
    model.train(data=DATASET, epochs=700, batch=32)
