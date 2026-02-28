# 垃圾分类/物体检测移动端 App

## 项目简介
本项目旨在开发一个移动端应用程序，利用深度学习模型（YOLO）识别垃圾并进行分类。

## 技术栈 (Technology Stack)

本项目采用端到端的前沿 AI 技术栈，涵盖模型训练、模型优化及多端部署。

### 1. 深度学习与模型训练
- **核心算法**: YOLOv11n (Ultralytics) - 选用 Nano 版本以平衡移动端速度与精度。
- **训练框架**: PyTorch, Python 3.10+
- **模型优化**: 
    - **Android 端**: TFLite (Float16 量化) - 减小模型体积，提升推理速度。
    - **Web 端**: ONNX (Opset 12) - 兼容 ONNX Runtime Web。
- **数据集**: 包含 12 类垃圾样本 (Battery, Biological, Glass, Cardboard, Clothes, Metal, Paper, Plastic, Shoes, Trash 等)。

### 2. Android 移动端应用
- **开发语言**: Kotlin
- **最低版本**: Android 8.0 (API Level 26)
- **核心库**:
    - **CameraX**: 用于高效的相机预览和图像分析。
    - **TensorFlow Lite Support**: 负责在 Android 设备上加载模型并进行推理。
    - **ViewBinding**: 更安全的 UI 组件绑定方式。
- **架构**: MVVM (Model-View-ViewModel) 简易实现，逻辑与 UI 分离。
- **构建工具**: Gradle 8.2

### 3. Web 端演示
- **技术**: HTML5, CSS3, Vanilla JavaScript (无重型框架依赖)
- **推理引擎**: ONNX Runtime Web (WASM Backend)
- **特点**: 纯前端推理，无需后端服务器，保护用户隐私。

### 4. 开发环境
- **IDE**: Android Studio Hedgehog / VS Code
- **版本控制**: Git

## 目录结构
- `dataset/`: 存放训练和测试数据
- `training/`: 训练脚本和模型权重
- `app/`: 移动端应用代码
- `models/`: 导出的轻量化模型 (tflite/onnx)

## 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 准备数据: (参考 `dataset/README.md`)
3. 运行训练: `python training/train.py`
