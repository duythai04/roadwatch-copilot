from collections import defaultdict

from ultralytics import YOLO

from roadwatch.core.types import Detection


class VehicleTracker:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.4
    ):
        # Load YOLO model
        self.model = YOLO(model_path)

        # Confidence threshold
        self.confidence = confidence

        # Lưu lịch sử vị trí theo Track ID
        self.track_history = defaultdict(list)

        # COCO classes cần theo dõi
        self.classes = [
            0,  # person
            1,  # bicycle
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
        ]

    def track(self, frame):
        """
        Detection + ByteTrack.

        Returns:
            List[Detection]
        """

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.classes,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]

        detections = []

        # Không có object
        if result.boxes is None:
            return detections

        # Duyệt qua các object
        for box in result.boxes:

            # -------------------------
            # Bounding box
            # -------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # -------------------------
            # Confidence
            # -------------------------

            confidence = float(box.conf[0])

            # -------------------------
            # Class
            # -------------------------

            class_id = int(box.cls[0])

            class_name = self.model.names[class_id]

            # -------------------------
            # Track ID
            # -------------------------

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            # -------------------------
            # Center point
            # -------------------------

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # -------------------------
            # Track history
            # -------------------------

            if track_id is not None:

                self.track_history[track_id].append(
                    (center_x, center_y)
                )

                # Chỉ giữ 30 điểm gần nhất
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

            # -------------------------
            # Detection object
            # -------------------------

            detection = Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                track_id=track_id
            )

            detections.append(detection)

        return detections

    def get_track_history(self):
        """
        Trả về lịch sử vị trí của các object.
        """

        return self.track_history