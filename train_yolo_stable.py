from ultralytics import YOLO
import torch


def main():
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5/dataset.yaml"
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_enhance/dataset.yaml"
    # data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_8classes_enhance/dataset.yaml"
    data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_8classes/dataset.yaml"
    
    model = YOLO("yolo11n-seg.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/segment",
        name="fengjiyepian_506_v5_8classes_v1",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        patience=50,
        verbose=True,
        amp=True,
        workers=4,
        close_mosaic=20,
        label_smoothing=0.0,
    )
    
    print(f"\nTraining completed. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
