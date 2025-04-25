import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # 加载配置文件
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在！")
        return
    
    # 使用UTF-8编码打开YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查必要的配置项
    if 'source' not in config:
        print("错误：配置文件中缺少 'source' 项，请指定输入视频或图像目录！")
        return
    
    # 设置默认值和从配置中获取值
    model_path = config.get('track_model', 'runs/detect/train_improved4/weights/best.pt')
    show = config.get('show', False)
    save = config.get('save', True)
    device = config.get('device', 'cpu')
    project = config.get('project', 'runs/track')
    name = config.get('track_name', 'exp')
    
    # 是否使用检测模式(而非跟踪模式)
    detection_mode = config.get('detection_mode', False)
    
    # 获取数据集配置文件路径
    dataset_yaml = config.get('data', 'D:/Clouddisk/Dropbox/01-Research/2_co-research/HT_Sui/disp_track/data/dataset.yaml')
    
    # 加载数据集配置以获取类别信息
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
            class_names = dataset_config.get('names', {})
            print(f"加载类别配置: {class_names}")
    else:
        class_names = {0: 'target'}  # 默认类别
        print(f"警告: 未找到数据集配置文件 {dataset_yaml}, 使用默认类别")
    
    # 降低默认检测阈值
    conf_threshold = config.get('conf', 0.05)  # 进一步降低阈值到0.05
    print(f"使用检测阈值: {conf_threshold}")
    
    # 打印模型路径确认
    print(f"加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 加载模型
    model = YOLO(model_path)
    
    # 打印诊断信息
    print(f"模型架构: {model.task}")
    print(f"输入图像尺寸: {model.model.args['imgsz'] if hasattr(model.model, 'args') else '未知'}")
    
    # 检查源是否存在
    if not os.path.exists(config['source']):
        print(f"错误: 源路径 {config['source']} 不存在!")
        return
    
    # 根据模式选择检测或跟踪
    if detection_mode:
        # 执行普通检测，指定类别名称
        results = model.predict(
            source=config['source'],
            conf=conf_threshold,
            iou=config.get('iou', 0.5),
            show=show,
            save=save,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            classes=list(class_names.keys()) if class_names else None,  # 指定类别ID
            verbose=True  # 显示详细信息
        )
        
        # 结果分析
        if len(results) > 0:
            print("\n检测结果分析:")
            for i, r in enumerate(results):
                boxes = r.boxes
                if len(boxes) > 0:
                    print(f"图片 {i+1}/{len(results)}: 检测到 {len(boxes)} 个目标")
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        cls_name = class_names.get(cls_id, f"类别{cls_id}")
                        print(f"  - {cls_name}: 置信度 {conf:.3f}, 坐标 {box.xyxy[0].tolist()}")
                else:
                    print(f"图片 {i+1}/{len(results)}: 未检测到目标")
        
        print(f"检测完成！结果已保存到 {os.path.join(project, name)}/")
    else:
        # 执行追踪
        results = model.track(
            source=config['source'],
            conf=conf_threshold,
            iou=config.get('iou', 0.5),
            show=show,
            save=save,
            device=device,
            tracker=config.get('tracker', 'bytetrack.yaml'),
            project=project,
            name=name,
            exist_ok=True,
        )
        print(f"追踪完成！结果已保存到 {os.path.join(project, name)}/")

if __name__ == "__main__":
    main()
