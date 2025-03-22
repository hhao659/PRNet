from ultralytics import YOLO

def validate(original_weights, quantized_weights):
    original_model = YOLO(original_weights)
    original_results = original_model.val(data="coco128.yaml")
    original_map = original_results.results_dict["map"]
    
    quantized_model = YOLO(quantized_weights)
    quantized_results = quantized_model.val(data="coco128.yaml")
    quantized_map = quantized_results.results_dict["map"]
    
    map_drop = original_map - quantized_map
    print(f"原始模型mAP: {original_map:.4f}")
    print(f"量化模型mAP: {quantized_map:.4f}")
    print(f"精度下降: {map_drop:.4f}")
    assert map_drop <= 0.03, "精度下降超过3%！需调整剪枝率或增加微调轮次"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, default="yolov11n.pt")
    parser.add_argument("--quantized", type=str, required=True)
    args = parser.parse_args()
    
    validate(args.original, args.quantized)