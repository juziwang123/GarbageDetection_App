from ultralytics import YOLO
import os

def export_model():
    # 1. 找到训练好的最佳权重文件
    # 注意：这里假设你的训练结果保存在默认路径，如果不同请手动修改
    # 通常在 runs/detect/GarbageDetection_App/training/yolov8n_garbage_mobile/weights/best.pt
    
    # 尝试自动寻找最近的 best.pt
    search_dir = "runs/detect"
    best_pt_path = None
    
    for root, dirs, files in os.walk(search_dir):
        if "best.pt" in files:
            # 找到一个 best.pt，优先找 garbage_mobile 相关的
            path = os.path.join(root, "best.pt")
            if "garbage" in path:
                best_pt_path = path
                break
            if best_pt_path is None: # 备用
                best_pt_path = path

    if not best_pt_path:
        print("Error: Could not find 'best.pt'. Please provide the path manually.")
        # best_pt_path = "path/to/your/best.pt" # 手动指定
        return

    print(f"Found weight file: {best_pt_path}")

    # 2. 加载模型
    model = YOLO(best_pt_path)

    # 3. 导出为 TFLite (及其他格式)
    # int8 量化通常用于极低功耗设备，但需要校准数据。
    # 这里我们先导出 float32 或 float16，兼容性更好。
    try:
        print("Exporting to TFLite (Float32)...")
        model.export(format='tflite') 
        
        # 如果需要更极致的压缩，可以开启 int8 (需要数据)
        # model.export(format='tflite', int8=True, data='GarbageDetection_App/training/garbage_data.yaml')
        
        print("Export successful!")
        print(f"TFLite model should be in: {os.path.dirname(best_pt_path)}")
        
    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    export_model()
