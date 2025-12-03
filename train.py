from ultralytics import YOLO
import os

# 设置环境变量，确保优先使用 GPU (如果有的话)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # ================= 配置区域 =================
    # 1. 模型选择
    # yolov8n.pt = 速度最快，适合轻量级部署 (推荐)
    # yolov8s.pt = 精度稍高，速度稍慢
    base_model = 'yolov8s.pt' 
    
    # 2. 数据集配置路径
    data_config = 'dataset/data.yaml'
    
    # 3. 训练参数
    epochs = 100        # 训练轮数 (建议 100-300)
    imgsz = 640         # 图片大小 (必须是 32 的倍数)
    batch_size = 16     # 每次处理几张图 (显存不够就改小，比如 8 或 4)
    project_name = 'SupOS_Train_Result' # 训练结果保存的文件夹名
    exp_name = 'exp1_defect_detection'  # 本次实验名
    # ===========================================

    print(f"正在加载基础模型: {base_model}...")
    # 第一次运行会自动从网络下载 yolov8n.pt
    model = YOLO(base_model) 

    print("🚀 开始训练...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name=exp_name,
        device=0,       # device=0 用显卡, device='cpu' 用CPU
        patience=20,    # 如果20轮精度没提升，提前停止
        save=True,      # 保存模型
        cache=False     # 如果内存不够大，设为 False
    )

    print("\n✅ 训练完成！")
    print(f"最优模型已保存至: {project_name}/{exp_name}/weights/best.pt")
    print("请将 best.pt 复制到你的检测程序中使用。")

if __name__ == '__main__':
    # Windows 下必须放在 if __name__ == '__main__': 之下运行
    main()