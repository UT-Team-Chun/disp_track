import os
import yaml
from ultralytics import YOLO

def main():
    # 加载配置文件
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在！")
        return
    
    # 使用UTF-8编码打开YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数据配置文件存在
    if not os.path.exists(config['data']):
        print(f"错误：数据集配置文件 {config['data']} 不存在！")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.join(config['project'], config['name']), exist_ok=True)
    
    # 加载模型
    model = YOLO(config['model'])
    
    # 训练模型
    train_args = {
        'data': config['data'],
        'epochs': config['epochs'],
        'imgsz': config['imgsz'],
        'batch': config['batch'],
        'device': '0',  # 强制使用CPU
        'project': config['project'],
        'name': config['name'],
        'amp': False,  # 禁用自动混合精度训练
    }
    
    # 添加可选配置
    if 'patience' in config:
        train_args['patience'] = config['patience']
    if 'workers' in config:
        train_args['workers'] = config['workers']
    
    # 执行训练
    model.train(**train_args)
    
    # 验证模型
    model.val()
    
    print(f"训练完成！模型已保存到 {os.path.join(config['project'], config['name'], 'weights/')}")

if __name__ == "__main__":
    main()
