from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11-VisDrone.engine")  # build a new model from YAML
    # Validate the model
    model.val(val=True, data='VisDrone.yaml', split='val', batch=1)
