import os
import cv2

def create_mp4(folder_path):
    files = os.listdir(folder_path)
    avi_files = [f for f in files if f.lower().endswith('.avi')]
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    if len(avi_files) == 1 and not jpg_files:
        avi_path = os.path.join(folder_path, avi_files[0])
        mp4_path = os.path.join(folder_path, 'output.mp4')
        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open .avi file: {avi_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        cap.release()
        out.release()
    elif not avi_files and jpg_files:
        jpg_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        first_img_path = os.path.join(folder_path, jpg_files[0])
        img = cv2.imread(first_img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {first_img_path}")
        height, width, _ = img.shape
        mp4_path = os.path.join(folder_path, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        for jpg in jpg_files:
            img_path = os.path.join(folder_path, jpg)
            frame = cv2.imread(img_path)
            if frame is not None:
                out.write(frame)
        out.release()
    else:
        raise ValueError("Folder must contain either exactly one .avi file or multiple .jpg files (no mixing or other cases).")