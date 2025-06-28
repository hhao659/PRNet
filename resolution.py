import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def resize_image(image_path, target_size=(640, 640), method='bilinear', keep_aspect_ratio=True):
    """
    调整图像分辨率
    
    Args:
        image_path: 输入图像路径
        target_size: 目标尺寸 (width, height)
        method: 插值方法 ('bilinear', 'nearest', 'bicubic', 'area')
        keep_aspect_ratio: 是否保持宽高比
    
    Returns:
        resized_image: 调整后的图像
        original_image: 原始图像
    """
    
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_height, original_width = original_image.shape[:2]
    
    print(f"原始图像尺寸: {original_width} x {original_height}")
    
    # 选择插值方法
    interpolation_methods = {
        'bilinear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }
    
    if method not in interpolation_methods:
        raise ValueError(f"不支持的插值方法: {method}")
    
    target_width, target_height = target_size
    
    if keep_aspect_ratio:
        # 计算保持宽高比的新尺寸
        aspect_ratio = original_width / original_height
        target_aspect_ratio = target_width / target_height
        
        if aspect_ratio > target_aspect_ratio:
            # 原图更宽，以宽度为准
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # 原图更高，以高度为准
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # 先调整到计算出的尺寸
        resized = cv2.resize(original_image, (new_width, new_height), 
                           interpolation=interpolation_methods[method])
        
        # 创建目标尺寸的黑色背景
        resized_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # 将调整后的图像放置在中心
        resized_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        print(f"保持宽高比，实际图像尺寸: {new_width} x {new_height}")
        print(f"填充后尺寸: {target_width} x {target_height}")
        
    else:
        # 直接拉伸到目标尺寸
        resized_image = cv2.resize(original_image, target_size, 
                                 interpolation=interpolation_methods[method])
        print(f"拉伸到目标尺寸: {target_width} x {target_height}")
    
    return resized_image, original_image

def compare_resolutions(image_path, target_sizes=[(640, 640), (416, 416), (320, 320)], 
                       methods=['bilinear'], keep_aspect_ratio=True):
    """
    对比不同分辨率的效果
    
    Args:
        image_path: 输入图像路径
        target_sizes: 目标尺寸列表
        methods: 插值方法列表
        keep_aspect_ratio: 是否保持宽高比
    """
    
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 计算子图布局
    num_methods = len(methods)
    num_sizes = len(target_sizes)
    total_plots = (num_sizes * num_methods) + 1  # +1 for original
    
    # 创建足够大的画布
    cols = min(4, total_plots)
    rows = (total_plots + cols - 1) // cols
    
    plt.figure(figsize=(5*cols, 4*rows))
    
    # 显示原始图像
    plt.subplot(rows, cols, 1)
    plt.imshow(original_image)
    plt.title(f'原始图像\n{original_image.shape[1]}x{original_image.shape[0]}')
    plt.axis('off')
    
    plot_idx = 2
    
    # 对每种方法和尺寸组合进行处理
    for method in methods:
        for target_size in target_sizes:
            try:
                resized_image, _ = resize_image(image_path, target_size, method, keep_aspect_ratio)
                
                plt.subplot(rows, cols, plot_idx)
                plt.imshow(resized_image)
                plt.title(f'{method.capitalize()}\n{target_size[0]}x{target_size[1]}')
                plt.axis('off')
                
                plot_idx += 1
                
            except Exception as e:
                print(f"处理 {method} {target_size} 时出错: {e}")
    
    plt.tight_layout()
    plt.show()

def batch_resize_images(input_folder, output_folder, target_size=(640, 640), 
                       method='bilinear', keep_aspect_ratio=True):
    """
    批量处理文件夹中的图像
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        target_size: 目标尺寸
        method: 插值方法
        keep_aspect_ratio: 是否保持宽高比
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                resized_image, _ = resize_image(input_path, target_size, method, keep_aspect_ratio)
                
                # 保存图像
                resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, resized_image_bgr)
                
                print(f"已处理: {filename}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 单张图像转换
    image_path = "AI-TOD-Select/P2476__1.0__600___1390.png"  # 替换为你的图像路径
    
    # 检查文件是否存在
    if os.path.exists(image_path):
        # 转换为640x640
        resized_img, original_img = resize_image(
            image_path, 
            target_size=(700, 700), 
            method='bilinear',
            keep_aspect_ratio=True
        )
        
        # 保存结果
        output_path = "resized_700.jpg"
        resized_img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, resized_img_bgr)
        print(f"已保存调整后的图像: {output_path}")
        
        # 可视化对比
        compare_resolutions(
            image_path,
            target_sizes=[(640, 640), (416, 416), (320, 320)],
            methods=['bilinear', 'nearest'],
            keep_aspect_ratio=True
        )
    else:
        print(f"图像文件不存在: {image_path}")
        print("请将 'your_image.jpg' 替换为实际的图像路径")
    
    # 示例2: 批量处理（取消注释使用）
    # batch_resize_images(
    #     input_folder="input_images",
    #     output_folder="output_images_640x640",
    #     target_size=(640, 640),
    #     method='bilinear',
    #     keep_aspect_ratio=True
    # )
