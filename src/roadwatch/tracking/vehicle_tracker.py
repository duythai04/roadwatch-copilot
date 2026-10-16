
from ultralytics import YOLO

from roadwatch.core.types import Detection


class VehicleTracker:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.4
    ):
        self.model = YOLO(model_path)

        self.confidence = confidence

        # COCO classes
        self.classes = [
            0,  # person
            1,  # bicycle
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
        ]

    def track(self, frame):

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

        # Không phát hiện object nào
        if result.boxes is None:
            return detections

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = self.model.names[class_id]

            # Track ID có thể chưa tồn tại ở một số frame
            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

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