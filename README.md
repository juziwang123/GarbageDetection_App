# 垃圾分类/物体检测移动端 App

## 项目简介
本项目旨在开发一个移动端应用程序，利用深度学习模型（YOLO）识别垃圾并进行分类。

## 技术栈
- **核心算法**: YOLOv8 / YOLOv11
- **机器学习框架**: PyTorch / TensorFlow Lite
- **移动端**: (待定: Android/iOS/Flutter)
- **关键技术**: 
    - 数据增强 (Data Augmentation)
    - 模型轻量化 (Model Quantization/Pruning)
    - 端侧部署 (Edge Deployment)

## 目录结构
- `dataset/`: 存放训练和测试数据
- `training/`: 训练脚本和模型权重
- `app/`: 移动端应用代码
- `models/`: 导出的轻量化模型 (tflite/onnx)

## 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 准备数据: (参考 `dataset/README.md`)
3. 运行训练: `python training/train.py`
