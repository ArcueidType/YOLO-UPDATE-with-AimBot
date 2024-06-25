from ultralytics import YOLO

DATASET = r'coco128.yaml'

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-tr3.yaml')  # transfer to deprecate some pretrained weights
    model = model.load('yolov8n.pt')
    model.train(data=DATASET, epochs=500, batch=32)
