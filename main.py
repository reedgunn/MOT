# https://docs.ultralytics.com/modes/train/#train-settings

from ultralytics import YOLO
from pathlib import Path
import os
from utils.visualize import create_visualization_video
import shutil
from ultralytics.utils.benchmarks import benchmark

RUNS_OUTPUT_PATH = Path("runs/detect")


def train(
    gpu_ids: list[int],
    training_run_name: str = "0"
) -> None:

    num_gpus = len(gpu_ids)
    batch_size_per_gpu = 53
    batch_size = num_gpus * batch_size_per_gpu

    training_run_name = f"train-{training_run_name}"

    training_run_output_path = RUNS_OUTPUT_PATH / training_run_name
    if os.path.exists(training_run_output_path):
        should_overwrite = input(f"Warning: you are about to overwrite the training run at '{training_run_output_path}' - would you like to proceed? [y|N] ")
        if should_overwrite != "y":
            exit()
        shutil.rmtree(training_run_output_path)

    model = YOLO(
        model="yolo12s.pt",
        verbose=True
    )

    model.train(
        data="VisDrone.yaml",  
        epochs=10,  
        patience=50,  
        batch=batch_size,  
        # batch=-1,
        cache="ram",  
        device=gpu_ids,
        # device=[1],
        workers=29,  
        name=training_run_name,
        cos_lr=True,  
        profile=True,
        plots=True,
        degrees=180.0,  
        translate=0.2,  
        scale=1.0,  
        shear=10.0,  
        perspective=0.001,  
        flipud=0.5,  
        mixup=0.1,  
        copy_paste=0.1
    )


def track(
    data_path: str,
    training_run_name: str = "0",
    weights_version: str = "best"
) -> None:

    data_path = Path(data_path)

    weights_path = RUNS_OUTPUT_PATH / f"train-{training_run_name}/weights/{weights_version}.pt"

    tracking_run_name = f"track-{training_run_name}_{weights_version}_{data_path.stem}"

    tracking_run_output_path = RUNS_OUTPUT_PATH / tracking_run_name
    if os.path.exists(tracking_run_output_path):
        should_proceed = input("You may have already ran this evaluation. Would you like to proceed? [y/N] ")
        if should_proceed != "y":
            exit()
        shutil.rmtree(tracking_run_output_path)

    model = YOLO(
        model=weights_path,
        verbose=True
    )

    results = model.track(
        source=data_path,
        persist=True,
        tracker="tracker.yaml",
        save_txt=True,
        name = tracking_run_name,
        conf=0.25,
        iou=0.5
    )
    
    print(results)
    
    create_visualization_video(results, tracking_run_output_path, 1, 1, 15)
    

def eval(
    training_run_name: str = "0",
    weights_version: str = "best"
) -> None:

    weights_path = RUNS_OUTPUT_PATH / f"train-{training_run_name}/weights/{weights_version}.pt"

    model = YOLO(
        model=weights_path,
        verbose=True
    )
    
    model.val(
        # data="coco8.yaml",
        # batch=53,
        # device=[1]
    )


def export_model(
    training_run_name: str = "0",
    weights_version: str = "best"
) -> None:
    # https://docs.ultralytics.com/modes/export/

    weights_path = RUNS_OUTPUT_PATH / f"train-{training_run_name}/weights/{weights_version}.pt"

    model = YOLO(
        model=weights_path,
        verbose=True
    )
    export_path = model.export(
        format="engine",
        int8=True,
        simplify=True,  # Simplify ONNX graph for better compatibility
        device=1
    )
    print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    # train(
    #     gpu_ids=list(range(1, 8))
    # )
    # eval()
    # track(
    #     data_path="/data/datasets/research-datasets/VisDrone/VisDrone2019-MOT-val/sequences/uav0000137_00458_v"
    # )
    # export_model()
    # benchmark(
    #     model="runs/detect/train-0/weights/best.pt",
    #     data="VisDrone.yaml",
    #     imgsz=640,
    #     int8=True,
    #     device=[1],
    #     format="engine",
    #     dynamic=True  # Enable dynamic batch sizing
    # )
    
    from ultralytics import YOLO
    import time

    model = YOLO("runs/detect/train-0/weights/best.engine")
  
    # Warmup
    for _ in range(10):
        model("datasets/VisDrone/images/val/0000026_03000_d_0000030.jpg", imgsz=640, device=1)

    # Measure inference time
    times = []
    for _ in range(100):  # Run multiple iterations for average
        start = time.time()
        model("datasets/VisDrone/images/val/0000026_03000_d_0000030.jpg", imgsz=640, device=1)
        times.append(time.time() - start)

    avg_time_ms = sum(times) / len(times) * 1000
    fps = 1000 / avg_time_ms
    print(f"Average inference time: {avg_time_ms:.2f} ms/im")
    print(f"FPS: {fps:.2f}")
    