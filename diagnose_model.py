import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import torch

def diagnose_model():
    """诊断模型检测失败的原因"""
    print("开始诊断模型...")
    
    # 强制使用CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_default_device('cpu')
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载数据集配置
    with open(config['data'], 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    # 检查模型文件是否存在
    model_path = config.get('track_model', 'runs/detect/train_fixed3/weights/best.pt')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
        
    # 检查测试数据目录是否存在
    test_dir = config.get('source', 'd:\\clouddisk\\googledrive\\0-work\\disp_track\\data\\test')
    if not os.path.exists(test_dir):
        print(f"错误: 测试目录 {test_dir} 不存在!")
        return
    
    # 加载模型 - 明确指定CPU设备
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 查看模型结构信息
    print(f"模型任务: {model.task}")
    print(f"模型类别: {model.names}")
    
    # 检查训练和测试图像
    print("\n对比训练和测试图像...")
    
    # 找到一个训练图像用于比较
    train_txt = dataset_config.get('train')
    if train_txt and os.path.exists(train_txt):
        with open(train_txt, 'r') as f:
            train_images = f.readlines()
        if train_images:
            train_img_path = train_images[0].strip()
            if os.path.exists(train_img_path):
                print(f"训练图像示例: {train_img_path}")
                train_img = cv2.imread(train_img_path)
                if train_img is not None:
                    print(f"训练图像尺寸: {train_img.shape}")
    
    # 获取一个测试图像
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    if test_images:
        test_img_path = test_images[0]
        print(f"测试图像示例: {test_img_path}")
        test_img = cv2.imread(test_img_path)
        if test_img is not None:
            print(f"测试图像尺寸: {test_img.shape}")
    
    # 尝试不同的置信度进行推理
    print("\n尝试不同置信度进行推理...")
    confidence_levels = [0.01, 0.001, 0.0001]
    for conf in confidence_levels:
        print(f"\n使用置信度阈值 {conf}:")
        try:
            results = model.predict(
                source=test_img_path,
                conf=conf,
                iou=0.01,  # 非常低的IOU阈值
                verbose=True,
                device='cpu'  # 明确指定CPU设备
            )
            
            if len(results) > 0:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    print(f"检测到 {len(boxes)} 个目标")
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0].item())
                        name = model.names[cls]
                        conf = box.conf[0].item()
                        print(f"  目标 {i+1}: 类别={name}, 置信度={conf:.4f}, 位置={box.xyxy[0].tolist()}")
                else:
                    print("未检测到任何目标")
        except Exception as e:
            print(f"推理出错: {str(e)}")
    
    # 尝试使用预训练模型
    print("\n尝试使用原始预训练模型进行比较...")
    try:
        pretrained_model = YOLO("yolov8n.pt")
        results = pretrained_model.predict(
            source=test_img_path,
            conf=0.25,
            verbose=True,
            device='cpu'  # 明确指定CPU设备
        )
        
        if len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) > 0:
                print(f"预训练模型检测到 {len(boxes)} 个目标")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0].item())
                    name = pretrained_model.names[cls]
                    conf = box.conf[0].item()
                    print(f"  目标 {i+1}: 类别={name}, 置信度={conf:.4f}, 位置={box.xyxy[0].tolist()}")
            else:
                print("预训练模型未检测到任何目标")
    except Exception as e:
        print(f"预训练模型推理出错: {str(e)}")
    
    # 检查模型训练日志
    print("\n检查模型训练历史...")
    try:
        model_dir = os.path.dirname(os.path.dirname(model_path))
        results_file = os.path.join(model_dir, 'results.csv')
        if os.path.exists(results_file):
            print(f"找到训练日志: {results_file}")
            # 读取前几行和最后几行以了解训练状态
            with open(results_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print("训练开始时指标:")
                    print(''.join(lines[:2]))
                    if len(lines) > 5:
                        print("训练结束时指标:")
                        print(''.join(lines[-3:]))
        else:
            print(f"未找到训练日志: {results_file}")
    except Exception as e:
        print(f"检查训练日志时出错: {str(e)}")
    
    # 建议下一步操作
    print("\n诊断完成!")
    print("\n建议:")
    print("1. 训练问题: 增加训练轮数至少50轮，确保训练过程正常完成")
    print("2. 数据问题: 确保测试图像与训练图像特征一致，标注质量高")
    print("3. 模型问题: 当前训练可能不充分，或目标特征不明显")
    print("4. CUDA兼容性问题: 您的PyTorch版本与CUDA不兼容，请坚持使用CPU进行训练和推理")
    print("5. 重新训练建议: 使用yolov8s.pt预训练模型，更多数据，更长训练时间")

if __name__ == "__main__":
    diagnose_model()
