import os
import glob
import shutil
from pathlib import Path

def check_dataset():
    """检查数据集完整性并修复图片与标注不匹配的问题"""
    # 设置路径
    dataset_dir = "data/yolo_dataset"
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    # 创建必要的目录
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs("data/fixed_dataset/images", exist_ok=True)
    os.makedirs("data/fixed_dataset/labels", exist_ok=True)
    
    # 获取所有图片和标签文件
    image_files = glob.glob(os.path.join(images_dir, "*.png")) + \
                  glob.glob(os.path.join(images_dir, "*.jpg"))
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    # 创建文件名到路径的映射
    label_map = {Path(label_path).stem: label_path for label_path in label_files}
    
    print(f"找到 {len(image_files)} 张图片和 {len(label_files)} 个标签文件")
    
    # 检查每个图片是否有对应的标签文件
    matched_pairs = 0
    unmatched_images = []
    
    for img_path in image_files:
        img_name = Path(img_path).stem
        
        if img_name in label_map:
            matched_pairs += 1
            
            # 复制匹配的图片和标签到固定数据集
            shutil.copy(img_path, f"data/fixed_dataset/images/{img_name}.png")
            shutil.copy(label_map[img_name], f"data/fixed_dataset/labels/{img_name}.txt")
            
        else:
            unmatched_images.append(img_name)
    
    print(f"匹配的图片和标注对: {matched_pairs}")
    print(f"未匹配的图片数量: {len(unmatched_images)}")
    if unmatched_images:
        print("前5个未匹配的图片:", unmatched_images[:5])
    
    # 创建训练和验证集文件列表
    if matched_pairs > 0:
        create_train_val_split("data/fixed_dataset")

def create_train_val_split(dataset_dir, val_ratio=0.2):
    """创建训练和验证集文件列表"""
    images_dir = os.path.join(dataset_dir, "images")
    image_files = [os.path.join("data/fixed_dataset/images", f) for f in os.listdir(images_dir)]
    
    if not image_files:
        print("错误: 没有找到匹配的图片文件，无法创建训练和验证集")
        return
    
    # 随机打乱
    import random
    random.shuffle(image_files)
    
    # 分割
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # 写入文件
    with open(os.path.join(dataset_dir, "train.txt"), 'w') as f:
        f.write("\n".join(train_files))
    
    with open(os.path.join(dataset_dir, "val.txt"), 'w') as f:
        f.write("\n".join(val_files))
    
    print(f"创建了训练集 ({len(train_files)}个文件) 和验证集 ({len(val_files)}个文件)")

if __name__ == "__main__":
    check_dataset()
