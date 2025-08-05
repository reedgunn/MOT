A real-time multi-object tracking model for a drone. The drone will have 1 camera (pointed downward) and fly around in the sky. For each object in each frame coming from the drone’s video feed, the model will output a 2D AABB, a class ID, and an instance ID. The drone will have an NVIDIA Jetson AGX Orin 64GB Module, and end-to-end inference will need to run at >=30 FPS. The goal is to achieve the highest mAP and HOTA.

tracking-by-detection

fast object detector: YOLO (probably from Ultralytics)

fine-tune on open-source datasets:
1. VisDrone
2. UAVDT
3. MOTChallenge

export model to TensorRT

metrics:
- FPS
- mAP
- HOTA
