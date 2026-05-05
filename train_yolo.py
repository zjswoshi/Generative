from ultralytics import YOLO
import torch


def main():
    data_yaml = "/home/cn/yolo/AnomalyAny/fengjiyepian_506_v5/fengjiyepian_506_v5_enhance/dataset.yaml"
    
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/detect",
        name="fengjiyepian_506_v5_enhance",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        patience=50,
        verbose=True,
        amp=True,
        workers=8,
    )
    
    print(f"\nTraining completed. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
