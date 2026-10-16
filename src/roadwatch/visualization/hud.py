import cv2
import numpy as np


class HUD:

    @staticmethod
    def draw_detections(frame, detections):
        """
        Vẽ bounding box, class, confidence và Track ID.
        """

        for detection in detections:

            # =========================
            # Bounding Box
            # =========================

            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 255, 0),
                2
            )

            # =========================
            # Label
            # =========================

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

            # =========================
            # Draw Label
            # =========================

            cv2.putText(
                frame,
                label,
                (
                    detection.x1,
                    max(20, detection.y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame

    @staticmethod
    def draw_tracks(frame, track_history):
        """
        Vẽ đường di chuyển (trajectory)
        của từng object dựa trên Track ID.
        """

        for track_id, points in track_history.items():

            # Cần ít nhất 2 điểm để vẽ đường
            if len(points) < 2:
                continue

            # Chuyển list thành numpy array
            points_array = np.array(
                points,
                dtype=np.int32
            ).reshape((-1, 1, 2))

            # Vẽ trajectory
            cv2.polylines(
                frame,
                [points_array],
                False,
                (255, 255, 0),
                2
            )

        return frame

    @staticmethod
    def draw_fps(frame, fps):
        """
        Hiển thị FPS.
        """

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