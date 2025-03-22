from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11-S2.yaml")  # 替换为你的模型配置文件
    
    # 加载本地预训练权重
    model.load("../runs/detect/train3/weights/best.pt")
    
    model.train(
        data="VisDrone.yaml",
        epochs=5,
        imgsz=640,
        batch=8,
        amp=True,        # 开启混合精度
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        project="runs/train",
        name="fp16_train",
    )