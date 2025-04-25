import os
import json
import glob
from pathlib import Path
from PIL import Image

def convert_labelme_to_yolo(json_dir, img_dir,output_dir, class_name="track_point"):
    """
    将LabelMe格式的json文件转换为YOLO格式的txt文件
    
    参数:
    json_dir: 包含LabelMe标注文件的目录
    output_dir: YOLO格式标注文件和图像的输出目录
    class_name: 类别名称
    """
    # 创建输出目录结构
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 获取所有json文件
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_file in json_files:
        try:
            # 读取json文件
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 获取图像路径和文件名
            img_name = Path(data['imagePath']).stem
            # 正确组合图像路径 - 将目录与文件名结合
            img_path = os.path.join(img_dir, f"{img_name}.png")
            
            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                print(f"警告: 找不到图像文件 {img_path}")
                # 尝试其他格式
                alt_path = os.path.join(img_dir, f"{img_name}.jpg")
                if os.path.exists(alt_path):
                    img_path = alt_path
                    print(f"找到替代图像: {alt_path}")
                else:
                    # 尝试根据JSON文件位置寻找图像
                    alternative_paths = [
                        os.path.join(os.path.dirname(json_file), f"{img_name}.png"),
                        os.path.join(os.path.dirname(json_file), f"{img_name}.jpg")
                    ]
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            img_path = alt_path
                            print(f"找到替代图像: {alt_path}")
                            break
                    else:
                        print(f"错误: 无法找到图像 {img_name}")
                        continue
            
            # 获取图像尺寸
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                
                # 复制图像到目标目录
                img.save(os.path.join(images_dir, f"{img_name}.png"))
            
            # 创建对应的txt文件
            txt_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            with open(txt_path, 'w') as f:
                # 处理所有标注的形状
                for shape in data.get('shapes', []):
                    if shape['shape_type'] == 'rectangle':
                        # LabelMe的矩形是由两个点定义的：左上角和右下角
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # 计算YOLO格式：中心点坐标和宽高（相对值）
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = abs(x2 - x1) / img_width
                        height = abs(y2 - y1) / img_height
                        
                        # 写入YOLO格式: <class_id> <x_center> <y_center> <width> <height>
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
            print(f"处理完成: {json_file} -> {txt_path}")
        except Exception as e:
            print(f"处理 {json_file} 时出错: {e}")

def create_dataset_splits(data_dir, split_ratio=0.8):
    """
    创建训练/验证数据集划分
    
    参数:
    data_dir: 包含images和labels子目录的数据目录
    split_ratio: 训练集比例
    """
    images_dir = os.path.join(data_dir, "images")
    
    # 获取所有图像文件
    image_files = [os.path.basename(f) for f in glob.glob(os.path.join(images_dir, "*.png"))]
    
    # 随机打乱
    import random
    random.shuffle(image_files)
    
    # 划分训练/验证
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # 创建train.txt和val.txt
    for name, files in [("train", train_files), ("val", val_files)]:
        with open(os.path.join(data_dir, f"{name}.txt"), 'w') as f:
            for img_file in files:
                f.write(f"{os.path.join(images_dir, img_file)}\n")
    
    print(f"数据集划分完成: 训练集 {len(train_files)} 张图片, 验证集 {len(val_files)} 张图片")

if __name__ == "__main__":
    # 示例用法
    json_dir = "data/raw/bbox"
    image_dir = "data/raw/images"
    output_dir = "data/yolo_dataset"
    
    # 转换数据格式
    convert_labelme_to_yolo(json_dir, image_dir, output_dir)
    
    # 创建数据集划分
    create_dataset_splits(output_dir)
