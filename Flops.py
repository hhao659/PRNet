import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from thop import profile, clever_format
#from ultralytics.nn.modules.idea import SliceSamp,SliceUpsamp,DSConv,PixelSliceConcat

class DSConv(nn.Module):  # EnhancedDepthwiseConv
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super(DSConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c1, c1*depth_multiplier, kernel_size=k, stride=s, padding=k//2, groups=c1, bias=False),
            nn.BatchNorm2d(c1 * depth_multiplier),
            nn.GELU() if act else nn.Identity(),
            nn.Conv2d(c1*depth_multiplier, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)

#class PixelSliceConcat(nn.Module):
#    def forward(self, x):
#        return torch.cat([
#            x[..., ::2, ::2],
#            x[..., 1::2, ::2],
#            x[..., ::2, 1::2],
#            x[..., 1::2, 1::2],
#        ], dim=1)

class ESSamp(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super(ESSamp, self).__init__()
        self.dsconv = DSConv(c1 * 4, c2, k=k, s=s, act=act,depth_multiplier=depth_multiplier)
        self.slices = nn.PixelUnshuffle(2)
        #self.slices = PixelSliceConcat()


    def forward(self, x):
        x = self.slices(x)
        return self.dsconv(x)


# 自定义 FLOPs 计算函数 for DSConv
# 直接在函数中实现 Conv2d 的 FLOPs 计算
def count_dsconv(m, x, y):
    x = x[0] # thop passes inputs as a tuple
    total_ops = 0

    # Helper function to calculate Conv2d FLOPs
    def calculate_conv2d_flops(conv_layer, input_tensor, output_tensor):
        # For grouped convolution, the number of operations is divided by the number of groups.
        # MACs = Cin/groups * Cout * Kh * Kw * Hout * Wout
        # FLOPs = 2 * MACs (typically)
        cin = conv_layer.in_channels
        cout = conv_layer.out_channels
        kh, kw = conv_layer.kernel_size
        groups = conv_layer.groups
        hout, wout = output_tensor.shape[2:]

        # Ensure correct calculation for depthwise conv (groups == cin == cout)
        if groups == cin and groups == cout:
             # Depthwise convolution calculation
             return 2 * (cin / groups) * (cout) * kh * kw * hout * wout # This simplifies to 2 * cin * kh * kw * hout * wout for depthwise
        else:
             # Standard or grouped convolution calculation
             return 2 * (cin / groups) * cout * kh * kw * hout * wout


    # Layer 1: Depthwise Conv
    # We need the output shape of this layer to calculate its FLOPs
    # Let's do a quick forward pass through the first part to get intermediate shape
    try:
         temp_output_l1 = m.block[0](x) # Output of Depthwise Conv
         total_ops += torch.DoubleTensor([calculate_conv2d_flops(m.block[0], x, temp_output_l1)])

         # Layer 2: BatchNorm (usually negligible FLOPs, but we included a simplified count before)
         # Keep the simplified count for consistency if desired
         if isinstance(m.block[1], nn.BatchNorm2d):
              temp_output_l2 = m.block[1](temp_output_l1) # Output after BN
              total_ops += torch.DoubleTensor([2 * m.block[1].num_features * temp_output_l2.shape[2] * temp_output_l2.shape[3]])

         # Layer 3: GELU or Identity (ignored for simplicity)
         # temp_output_l3 = m.block[2](temp_output_l2) # Output after Activation

         # Layer 4: Pointwise Conv
         # Use the output shape of the DSConv block (y) and input channels of the pointwise conv
         # Input channels of pointwise conv is the output channels of the depthwise conv and BN
         # which is m.block[3].in_channels
         input_pointwise = temp_output_l2 # Input to pointwise conv is output of BN
         total_ops += torch.DoubleTensor([calculate_conv2d_flops(m.block[3], input_pointwise, y)]) # y is the final output of DSConv


         # Layer 5: BatchNorm (after pointwise conv)
         if isinstance(m.block[4], nn.BatchNorm2d):
             # The output shape of this BN is the same as the final output y
              total_ops += torch.DoubleTensor([2 * m.block[4].num_features * y.shape[2] * y.shape[3]])

         # Layer 6: GELU or Identity (ignored for simplicity)

    except Exception as e:
        print(f"Error during DSConv FLOPs calculation: {e}")
        print("Please check the intermediate tensor shapes.")
        # Fallback or raise error if calculation fails
        raise e


    m.total_ops += total_ops


def count_pixel_slice_concat(m, x, y):
    m.total_ops += torch.DoubleTensor([0])

def count_slicesamp(m, x, y):
     # thop will traverse submodules
     pass

def count_sliceupsamp(m, x, y):
    # thop will traverse submodules
    pass


# 定义自定义操作字典
custom_ops = {
    DSConv: count_dsconv,
    #PixelSliceConcat: count_pixel_slice_concat,
    # SliceSamp 和 SliceUpsamp 会自动调用其子模块的计数器
}

# 指定您的模型 YAML 文件路径
model_yaml_path = '/home/i/PRNet/ultralytics/cfg/models/11/yolo11l-PRNet.yaml'

# 指定用于 FLOPs 计算的图像尺寸
# 这应该与您训练时使用的 imgsz 参数一致，例如 640
input_img_size = 1024
input_tensor = torch.randn(1, 3, input_img_size, input_img_size) # batch size 为 1

try:
    # 加载模型结构 (不加载权重)
    # Ultralytics 的 YOLO 类可以直接加载 yaml 文件
    model = YOLO(model_yaml_path).model # 获取底层的 nn.Module

    # Profile 模型
    # 将模型移到 CPU 进行 profiling，以避免潜在的 GPU 相关问题
    model.cpu()
    flops, params = profile(model, inputs=(input_tensor.cpu(),), custom_ops=custom_ops, verbose=False)

    # 转换为可读格式并打印
    # thop 计算的是 MACs，通常转化为 FLOPs 乘以 2
    gflops = flops * 2 / 1E9
    mparams = params / 1E6

    print(f"Model: {model_yaml_path}")
    print(f"Input Image Size: {input_img_size}x{input_img_size}")
    print(f"Estimated FLOPs: {gflops:.3f} G")
    print(f"Parameters: {mparams:.3f} M")

except Exception as e:
    print(f"An error occurred during profiling: {e}")
    print("Please ensure your custom modules are correctly defined and accessible in the script.")
    print("Also, double-check the model_yaml_path and input_img_size.")
    print("If the error persists, there might be an issue within the custom counting function or module definition.")
