A multi-object tracking model for a drone. The drone will have 1 camera (pointed downward) and fly around in the sky. For each object in each frame coming from the drone’s video feed, the model will output a 2D AABB, a class id, and an instance id.


tracking-by-detection

fine-tune on open-source datasets:
1. VisDrone
2. UAVDT

metrics:
- HOTA
- FPS

fast object detector: YOLO (https://docs.ultralytics.com/modes/track/)


the best metric for real-time multi-object tracking models: HOTA
