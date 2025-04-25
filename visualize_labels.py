import os
import cv2
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt

def visualize_labels():
    """可视化检查标签与图像的对应关系"""
    dataset_dir = "data/fixed_dataset"
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    # 获取图像和标签文件
    image_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    
    if not image_files:
        print(f"错误: 未找到图像文件在 {images_dir}")
        return
    
    # 创建输出目录
    output_dir = "visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 类别名称
    class_names = {
        0: "target",
        1: "p",
        2: "sp"
    }
    
    # 设置不同类别的颜色
    colors = {
        0: (0, 255, 0),   # 绿色
        1: (0, 0, 255),   # 红色
        2: (255, 0, 0)    # 蓝色
    }
    
    print(f"处理 {len(image_files)} 张图像...")
    
    for img_path in image_files[:10]:  # 仅处理前10张图像
        img_name = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        height, width, _ = img.shape
        
        # 检查标签文件是否存在
        if os.path.exists(label_path):
            # 读取标签
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 在图像上绘制标签
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    
                    # 计算左上角和右下角的坐标
                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)
                    
                    # 绘制边界框
                    color = colors.get(class_id, (0, 255, 255))  # 默认黄色
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加类别标签
                    class_name = class_names.get(class_id, f"类别{class_id}")
                    cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存带标签的图像
            output_path = os.path.join(output_dir, f"{img_name}_labeled.jpg")
            cv2.imwrite(output_path, img)
            print(f"已保存: {output_path}")
        else:
            print(f"标签文件不存在: {label_path}")
    
    print(f"可视化完成! 结果保存在 {output_dir} 目录")

if __name__ == "__main__":
    visualize_labels()
