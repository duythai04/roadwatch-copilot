import cv2
import numpy as np

class HUD:

    @staticmethod
    def draw_detections(frame, detections):

        for detection in detections:

            # Bounding box
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 255, 0),
                2
            )

            # Label
            if detection.track_id is not None:
                label = (
                    f"{detection.class_name} "
                    f"ID:{detection.track_id} "
                    f"{detection.confidence:.2f}"
                )
            else:
                label = (
                    f"{detection.class_name} "
                    f"{detection.confidence:.2f}"
                )

            cv2.putText(
                frame,
                label,
                (detection.x1, max(20, detection.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame

    @staticmethod
    def draw_fps(frame, fps):

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return frame

    