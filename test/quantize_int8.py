import torch
from ultralytics import YOLO

def quantize_model(pruned_weights, calib_data_dir):
    model = YOLO(pruned_weights).model
    model.eval()
    
    # 准备校准数据（示例）
    calibration_data = [torch.randn(1, 3, 640, 640) for _ in range(100)]
    
    # 量化配置
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    prepared_model = torch.quantization.prepare(model)
    
    # 校准
    for data in calibration_data:
        prepared_model(data)
    
    # 转换为INT8
    quantized_model = torch.quantization.convert(prepared_model)
    quantized_path = pruned_weights.replace(".pt", "_int8.pt")
    torch.jit.save(torch.jit.script(quantized_model), quantized_path)
    print(f"量化模型已保存至: {quantized_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--calib_dir", type=str, default="calib_data/")
    args = parser.parse_args()
    
    quantize_model(args.weights, args.calib_dir)