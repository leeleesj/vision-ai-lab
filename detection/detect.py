from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

if __name__ == '__main__':
    model = load_model("{YOUR_MODEL_PATH}")
    model.predict("../assets/bus.jpg", save=True, imgsz=320, conf=0.5)