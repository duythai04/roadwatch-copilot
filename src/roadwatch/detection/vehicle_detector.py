from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence


        self.classes = [
            0,  # person
            1,  # bicycle
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
        ]

    def detect(self, frame):
        results = self.model(
            frame,
            classes=self.classes,
            conf=self.confidence,
            verbose=False
        )

        return results[0]