import json
import os
from PIL import Image
from tqdm import tqdm

def convert_ai2yolo(ann_file, dataset_dir, mode='test'):
    print(f"\nProcessing {mode} dataset...")
    
    # 构建目录路径
    images_dir = os.path.join(dataset_dir, mode, 'images')
    labels_dir = os.path.join(dataset_dir, mode, 'labels')
    
    # 确保标签目录存在
    os.makedirs(labels_dir, exist_ok=True)
    
    # 读取AI-TOD的标注文件
    try:
        with open(ann_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file {ann_file} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {ann_file}!")
        return
    
    # 清理已存在的标签文件
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            os.remove(os.path.join(labels_dir, file))
    
    # 获取图像信息
    image_info = {}
    for img in data['images']:
        image_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # 创建图像ID到标注的映射
    image_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # 处理每个图像
    processed = 0
    skipped = 0
    empty_labels = 0
    
    # 处理所有图像，即使没有标注
    for img_id, img_data in tqdm(image_info.items(), desc=f"Converting {mode} annotations"):
        img_filename = img_data['file_name']
        img_path = os.path.join(images_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_filename} not found, skipping...")
            skipped += 1
            continue
            
        try:
            # 使用图像信息中的尺寸，而不是打开图像
            img_w = img_data['width']
            img_h = img_data['height']
            
            # 确保尺寸有效
            if img_w <= 0 or img_h <= 0:
                raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")
                
        except Exception as e:
            print(f"Error processing image {img_filename}: {str(e)}")
            skipped += 1
            continue
            
        # YOLO格式标签路径 - 使用与图像相同的基础名称
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        try:
            with open(label_path, 'w') as f:
                if img_id in image_annotations:
                    for ann in image_annotations[img_id]:
                        # 解析bbox [x, y, width, height]
                        x, y, w, h = ann['bbox']
                        
                        # 确保bbox有效
                        if w <= 0 or h <= 0:
                            continue
                            
                        # 计算归一化后的中心坐标和宽高
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                        
                        # 确保值在0-1范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        w_norm = max(0, min(1, w_norm))
                        h_norm = max(0, min(1, h_norm))
                        
                        # 类别ID从0开始
                        class_id = ann['category_id'] - 1
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                else:
                    # 创建空标签文件
                    empty_labels += 1
                    
            processed += 1
        except Exception as e:
            print(f"Error writing label file for image {img_filename}: {str(e)}")
            skipped += 1
            continue
    
    print(f"\n{mode} dataset conversion complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Empty label files: {empty_labels}")
    print(f"Skipped: {skipped} images")
    print(f"Labels saved to: {labels_dir}")

# 示例调用
if __name__ == "__main__":
    # 数据集根目录
    dataset_dir = '/data/home/zph/TFNet/datasets/AI-TOD'
    
    # 只处理test数据集
    ann_file = 'datasets/AI-TOD/annotations/aitod_test_v1.json'
    
    if not os.path.exists(ann_file):
        print(f"Error: Test annotation file {ann_file} not found!")
    else:
        convert_ai2yolo(ann_file, dataset_dir, 'test')
