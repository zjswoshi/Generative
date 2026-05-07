from ultralytics import YOLO
import torch


def main():
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5/dataset.yaml"
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_enhance/dataset.yaml"
    data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_8classes_enhance/dataset.yaml"
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_8classes/dataset.yaml"
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian260310_v17/fengjiyepian_260310_v17.yaml"
    
    model = YOLO("yolo11n-seg.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        device="cuda:1" if torch.cuda.is_available() else "cpu",
        project="runs/segment",
        name="fengjiyepian_506_v5_8classes_enhance",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        box=8.0,  # 稍微提高边框损失
        cls=0.8,  # 提高分类损失（27类需要更好的分类能力）
        dfl=1.5,  # 保持分布焦点损失
        hsv_h=0.01,  # 降低色相变化
        hsv_s=0.5,  # 降低饱和度变化
        hsv_v=0.3,  # 降低亮度变化
        degrees=10.0,  # 增加旋转范围
        translate=0.15,  # 增加平移
        scale=0.6,  # 增加尺度变化
        shear=5.0,  # 增加剪切
        flipud=0.3,  # 降低垂直翻转（叶片通常是水平方向）
        fliplr=0.5,  # 保持水平翻转
        mosaic=1.0,  # 保持 mosaic 增强
        mixup=0.2,  # 增加 mixup 到 0.2
        copy_paste=0.1,  # 启用 copy_paste
        patience=30,
        verbose=True,
        save_period=10,
        amp=True,
        workers=4,
        close_mosaic=20,
        label_smoothing=0.0,
    )
    
    print(f"\nTraining completed. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
