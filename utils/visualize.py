from typing import List
from pathlib import Path

from ultralytics.engine.results import Results
import cv2


def create_visualization_video(
    results_list: List[Results],
    output_path: str = "output.mp4",
    line_width: int = None,
    font_size: float = None,
    fps: int = 30
) -> None:
    height, width = results_list[0].orig_shape
    output_path = Path(output_path) / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    frame_count = 0
    for result in results_list:
        annotated_img = result.plot(line_width=line_width, font_size=font_size)
        video_writer.write(annotated_img)
        frame_count += 1
    video_writer.release()
