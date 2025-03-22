from ultralytics import YOLO

# 加载原始模型
original_model = YOLO("../yolo11n.pt")
original_params = sum(p.numel() for p in original_model.model.parameters())

pruned_model = YOLO("../yolo11n.pt")
pruned_model.prune(amount=0.5)

# 加载剪枝后模型
pruned_model = YOLO("../yolo11n.pt")
pruned_params = sum(p.numel() for p in pruned_model.model.parameters())

print(f"原始参数量: {original_params / 1e6:.2f}M")
print(f"剪枝后参数量: {pruned_params / 1e6:.2f}M")  