import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path
import warnings
from torch.jit import TracerWarning

warnings.filterwarnings("ignore", category=TracerWarning)

def preprocess_image(image_path, input_shape=(640, 640)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, input_shape)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    input_tensor = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
    return input_tensor

def run_onnx_inference(onnx_path, image_path):
    print(f"[INFO] Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name

    input_tensor = preprocess_image(image_path).astype(np.float32)

    # Warm-up
    for _ in range(5):
        session.run(None, {input_name: input_tensor})

    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    fps = 1 / avg_time
    print(f"[✓] Average inference time: {avg_time * 1000:.2f} ms")
    print(f"[✓] FPS: {fps:.2f}")
    print(f"[✓] Output shapes:")
    for i, out in enumerate(outputs):
        print(f"    Output {i}: shape={out.shape}")

def main():
    onnx_model_path = Path("abe_onnx.onnx")  # <-- Your ONNX model
    test_image_path = Path("0000022_01251_d_0000007.jpg")  # <-- Your image

    run_onnx_inference(onnx_model_path, test_image_path)

if __name__ == "__main__":
    main()
