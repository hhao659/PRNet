from ultralytics import YOLO
import torch
from torch import nn
import os

class PRUNE():
    def __init__(self):
        self.threshold = None

    def get_threshold(self, model, factor=0.8):
        """
        计算剪枝阈值，遍历所有 BatchNorm 层的 `gamma` 权重，确定剪枝的最小值
        """
        ws = []
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                ws.append(m.weight.abs().detach())

        if not ws:
            raise ValueError("未找到 BatchNorm2d 层，可能是错误的模型结构")

        ws = torch.cat(ws)
        self.threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
        print(f"剪枝阈值设定为：{self.threshold:.6f}")

    def prune_conv(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """
        剪枝 Conv + BatchNorm 组合
        """
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            return

        gamma = bn.weight.data.abs()
        keep_idxs = torch.where(gamma >= self.threshold)[0]

        if len(keep_idxs) < 8:
            keep_idxs = torch.argsort(gamma, descending=True)[:8]  # 至少保留 8 个通道

        # 更新 BatchNorm
        bn.weight.data = bn.weight.data[keep_idxs]
        bn.bias.data = bn.bias.data[keep_idxs]
        bn.running_mean.data = bn.running_mean.data[keep_idxs]
        bn.running_var.data = bn.running_var.data[keep_idxs]
        bn.num_features = len(keep_idxs)

        # 更新 Conv
        conv.weight.data = conv.weight.data[keep_idxs, :, :, :]
        conv.out_channels = len(keep_idxs)
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[keep_idxs]

        print(f"剪枝 Conv {conv.out_channels} 个通道，保留比例: {len(keep_idxs) / gamma.numel():.2%}")

    def prune_c3k2(self, module):
        """
        针对 C3k2 结构剪枝
        """
        if hasattr(module, 'cv1') and hasattr(module, 'cv2'):
            print(f"剪枝 C3k2 结构: {module}")
            self.prune_conv(module.cv1.conv, module.cv1.bn)
            self.prune_conv(module.cv2.conv, module.cv2.bn)

    def prune_model(self, model):
        """
        遍历模型的所有层，进行剪枝
        """
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                # 找下一个 BatchNorm 层
                next_bn = None
                for next_name, next_layer in model.named_modules():
                    if name in next_name and isinstance(next_layer, nn.BatchNorm2d):
                        next_bn = next_layer
                        break
                if next_bn:
                    print(f"剪枝 {name}")
                    self.prune_conv(layer, next_bn)
            elif 'C3k2' in str(type(layer)):  # 兼容 C3k2 结构
                self.prune_c3k2(layer)

    def prune_detect(self, model):
        """
        针对 YOLO 检测头剪枝，确保 box、cls、proto 分支一致
        """
        for name, layer in model.named_modules():
            if "detect" in name.lower() and isinstance(layer, nn.Conv2d):
                print(f"剪枝检测头: {name}")
                self.prune_conv(layer, layer)

def do_pruning(model_path, save_path, prune_ratio=0.65):
    """
    加载模型，执行剪枝，并保存
    """
    pruning = PRUNE()
    yolo = YOLO(model_path)

    # 计算剪枝阈值
    pruning.get_threshold(yolo.model, prune_ratio)

    # 剪枝所有 Conv + BN
    pruning.prune_model(yolo.model)

    # 剪枝检测头
    pruning.prune_detect(yolo.model)

    # 重新训练准备
    for name, p in yolo.model.named_parameters():
        p.requires_grad = True

    # 验证剪枝后效果
    yolo.val(data='data.yaml', batch=2, device=0, workers=0)

    # 保存剪枝后的模型
    torch.save(yolo.ckpt, save_path)
    print(f"剪枝后模型已保存至 {save_path}")

if __name__ == "__main__":
    model_path = "../yolo11n.pt"
    save_path = "../last_prune.pt"
    do_pruning(model_path, save_path, prune_ratio=0.5)
