from ultralytics import YOLO
from pathlib import Path
import os
import shutil
from typing import List
from utils.visualize import create_visualization_video
import time

RUNS_OUTPUT_PATH = Path("runs/detect")


def train(
    gpu_ids: List[int],
    training_run_name: str = "0"
) -> None:
    device = gpu_ids[0] if gpu_ids else 0  # Use first GPU or CPU fallback

    batch_size_per_gpu = 53
    batch_size = batch_size_per_gpu  # single GPU training for v8 API

    training_run_name = f"train-{training_run_name}"

    training_run_output_path = RUNS_OUTPUT_PATH / training_run_name
    if os.path.exists(training_run_output_path):
        should_overwrite = input(
            f"Warning: you are about to overwrite the training run at '{training_run_output_path}' - would you like to proceed? [y|N] "
        )
        if should_overwrite != "y":
            exit()
        shutil.rmtree(training_run_output_path)

    model = YOLO("yolo12s.pt", verbose=True)

    model.train(
        data="VisDrone.yaml",
        epochs=10,
        patience=50,
        batch=batch_size,
        cache="ram",
        device=device,
        workers=8,  # adjust depending on CPU cores
        name=training_run_name,
        cosine_lr=True,
        profile=True,
        plots=True,
        degrees=180.0,
        translate=0.2,
        scale=1.0,
        shear=10.0,
        perspective=0.001,
        flipud=0.5,
        mixup=0.1,
        copy_paste=0.1,
    )


def track(
    data_path: str,
    training_run_name: str = "0",
    weights_version: str = "best",
    device=0,
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

    model = YOLO(weights_path, verbose=True)

    results = model.track(
        source=data_path,
        persist=True,
        tracker="tracker.yaml",
        save_txt=True,
        name=tracking_run_name,
        conf=0.25,
        iou=0.5,
        device=device,
    )

    print(results)

    create_visualization_video(results, tracking_run_output_path, 1, 1, 15)


def eval(
    training_run_name: str = "0",
    weights_version: str = "best",
    device=0,
) -> None:
    weights_path = RUNS_OUTPUT_PATH / f"train-{training_run_name}/weights/{weights_version}.pt"
    model = YOLO(weights_path, verbose=True)

    model.val(
        # data="coco8.yaml",
        # batch=53,
        # device=device
    )


def export_model(
    training_run_name: str = "0",
    weights_version: str = "best",
    device=0,
) -> None:
    weights_path = RUNS_OUTPUT_PATH / f"train-{training_run_name}/weights/{weights_version}.pt"
    model = YOLO(weights_path, verbose=True)

    export_path = model.export(
        format="engine",
        int8=True,
        simplify=True,
        device=device,
    )
    print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    device = 0  # change to the GPU index or "cuda:0" string if needed

    # Uncomment and adjust calls below as needed:

    # train(
    #     gpu_ids=[device],
    #     training_run_name="0"
    # )

    # eval(
    #     training_run_name="0",
    #     device=device
    # )

    # track(
    #     data_path="/data/datasets/research-datasets/VisDrone/VisDrone2019-MOT-val/sequences/uav0000137_00458_v_0000030",
    #     training_run_name="0",
    #     device=device
    # )

    # export_model(
    #     training_run_name="0",
    #     device=device
    # )

    model = YOLO("yolov8n.pt")

    # Warmup
    for _ in range(10):
        model("0000022_01251_d_0000007.jpg", imgsz=640, device=device)

    # Measure inference time
    times = []
    for _ in range(100):
        start = time.time()
        model("0000022_01251_d_0000007.jpg", imgsz=640, device=device)
        times.append(time.time() - start)

    avg_time_ms = sum(times) / len(times) * 1000
    fps = 1000 / avg_time_ms
    print(f"Average inference time: {avg_time_ms:.2f} ms/im")
    print(f"FPS: {fps:.2f}")
