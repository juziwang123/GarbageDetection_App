from ultralytics import YOLO
import os

def train_model():
    # 检查是否有 GPU
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载预训练模型 (YOLOv8n 是 nano 版本，适合移动端)
    model = YOLO('yolov8n.pt') 

    # 训练参数设置
    # 注意: data 参数需要指向一个 data.yaml 文件，定义数据集路径
    # epochs: 训练轮数
    # imgsz: 输入图像大小
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_root = os.path.join(project_root, 'dataset', 'yolo_format')

    # 动态生成一个临时的 yaml 文件，使用正斜杠且处理好编码
    # 这样可以规避中文路径解析乱码的问题
    import yaml
    
    # 读取原始 yaml 配置（主要是 names）
    with open(os.path.join(current_dir, 'garbage_data.yaml'), 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    # 强制覆盖路径配置为绝对路径
    data_config['path'] = dataset_root.replace('\\', '/')
    data_config['train'] = 'train/images'
    data_config['val'] = 'val/images'
    
    # 写入临时文件
    temp_yaml_path = os.path.join(current_dir, 'temp_garbage_config.yaml')
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True)

    print(f"Dataset config created at: {temp_yaml_path}")
    print(f"Dataset root set to: {data_config['path']}")

    try:
        # Load a model
        # YOLOv11 nano (yolo11n.pt) - Update ultralytics package if needed: pip install -U ultralytics
        model = YOLO('yolo11n.pt')  

        # Train the model
        results = model.train(
            data=temp_yaml_path, 
            epochs=50, 
            imgsz=224, # Smaller image size for mobile
            batch=16,
            device=device,
            project='GarbageDetection_App/training',
            name='yolov8n_garbage_mobile',
            exist_ok=True,
            
            # === 数据增强配置 (Data Augmentation) ===
            augment=True,      # 启用默认增强
            hsv_h=0.015,       # 色调变化 (Hue)
            hsv_s=0.7,         # 饱和度变化 (Saturation) - 针对不同颜色的垃圾
            hsv_v=0.4,         # 亮度变化 (Value) - 适应不同光照
            degrees=10.0,      # 随机旋转 +/- 10度
            translate=0.1,     # 随机平移
            scale=0.5,         # 随机缩放 (0.5 ~ 1.5 倍)
            shear=0.0,         # 剪切 (保持为0，垃圾通常不变形)
            flipud=0.0,        # 上下翻转概率 (通常不需要，除非垃圾是倒着的)
            fliplr=0.5,        # 左右翻转概率 (50%)
            mosaic=1.0,        # 马赛克增强概率 (1.0 = always on) - 非常有效！
            mixup=0.1,         # Mixup 增强概率 (混合两张图)
        )
        
        # 导出为 TFLite 格式，用于移动端部署
        success = model.export(format='tflite')
        print(f"Model exported: {success}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # 确保在当前目录下运行，或者调整路径
    train_model()
