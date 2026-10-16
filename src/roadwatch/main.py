import cv2

from roadwatch.core.video_reader import VideoReader
from roadwatch.detection.vehicle_detector import VehicleDetector


def main():
    video_path = "data/raw/videos/test.mp4"

    reader = VideoReader(video_path)
    detector = VehicleDetector()

    print("FPS:", reader.get_fps())
    print("Width:", reader.get_width())
    print("Height:", reader.get_height())

    while True:
        ret, frame = reader.read()

        if not ret:
            break

        # YOLO Detection
        result = detector.detect(frame)

        annotated_frame = result.plot()

        cv2.imshow(
            "RoadWatch Copilot - Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()