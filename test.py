from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/Ti-FPN_AI-TOD/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("AI-TOD-Select/*.png",
              save=True, imgsz=640, show_labels=False, show_conf=False, line_width=2, max_det=500)
