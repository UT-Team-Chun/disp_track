import os
import json
import glob
from pathlib import Path

def convert_labelme_to_yolo():
    """将labelme标注转换为YOLO格式"""
    # 设置路径
    json_dir = "data/raw/bbox"  # JSON标注文件目录
    output_dir = "data/yolo_dataset/labels"  # 输出YOLO格式标签的目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    # 类别映射 - 根据实际情况调整
    class_map = {"target": 0}  # 将labelme中的类别名称映射到数字ID
    
    processed_count = 0
    error_count = 0
    
    for json_path in json_files:
        try:
            # 获取图像基本名称
            base_name = Path(json_path).stem
            
            # 转换JSON到YOLO格式
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像尺寸
            img_width = data.get('imageWidth', 0)
            img_height = data.get('imageHeight', 0)
            
            if img_width == 0 or img_height == 0:
                print(f"警告: {base_name}.json 缺少图像尺寸信息")
                continue
            
            # 创建YOLO格式的输出文件
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            
            with open(output_path, 'w') as f:
                if 'shapes' in data:
                    for shape in data['shapes']:
                        if 'label' in shape and 'points' in shape:
                            label = shape['label']
                            points = shape['points']
                            
                            # 检查标签是否在类别映射中
                            if label not in class_map:
                                print(f"警告: 未知类别 '{label}' 在文件 {base_name}.json 中")
                                class_map[label] = len(class_map)  # 自动添加新类别
                            
                            # 处理矩形 (bbox)
                            if shape['shape_type'] == 'rectangle' and len(points) == 2:
                                x1, y1 = points[0]
                                x2, y2 = points[1]
                                
                                # 计算YOLO格式的中心点和宽高
                                x_center = (x1 + x2) / 2 / img_width
                                y_center = (y1 + y2) / 2 / img_height
                                width = abs(x2 - x1) / img_width
                                height = abs(y2 - y1) / img_height
                                
                                # 获取类别ID
                                class_id = class_map[label]
                                
                                # 写入YOLO格式: <class_id> <x_center> <y_center> <width> <height>
                                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理 {json_path} 时出错: {str(e)}")
            error_count += 1
    
    print(f"处理完成! 成功转换 {processed_count} 个文件, 失败 {error_count} 个")
    print(f"最终类别映射: {class_map}")
    
    # 更新dataset.yaml中的类别
    update_dataset_yaml(class_map)

def update_dataset_yaml(class_map):
    """更新dataset.yaml文件中的类别信息"""
    yaml_path = "data/dataset.yaml"
    
    # 准备类别名称
    names_str = ""
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        names_str += f"  {idx}: {name}\n"
    
    # 创建或更新YAML文件
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("# 数据集配置\n")
        f.write("path: ./data/yolo_dataset\n")
        f.write("train: ./train.txt\n")
        f.write("val: ./val.txt\n\n")
        f.write("# 类别配置\n")
        f.write(f"nc: {len(class_map)}  # 类别数量\n")
        f.write("names:\n")
        f.write(names_str)

if __name__ == "__main__":
    convert_labelme_to_yolo()
