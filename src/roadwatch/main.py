import cv2

from roadwatch.core.video_reader import VideoReader
from roadwatch.tracking.vehicle_tracker import VehicleTracker
from roadwatch.visualization.hud import HUD
from roadwatch.utils.fps import FPSCounter
from roadwatch.lane.lane_detector import LaneDetector


def main():

    # =========================
    # 1. Video path
    # =========================

    video_path = "data/raw/videos/test.mp4"

    # =========================
    # 2. Initialize modules
    # =========================

    # Video reader
    reader = VideoReader(video_path)

    # YOLO + ByteTrack
    tracker = VehicleTracker(
        model_path="yolo11n.pt",
        confidence=0.4
    )

    # Lane detector
    lane_detector = LaneDetector()

    # HUD
    hud = HUD()

    # FPS counter
    fps_counter = FPSCounter()

    # =========================
    # 3. Video information
    # =========================

    print("Video FPS:", reader.get_fps())
    print("Width:", reader.get_width())
    print("Height:", reader.get_height())

    # =========================
    # 4. Main loop
    # =========================

    while True:

        # -------------------------
        # Read frame
        # -------------------------

        ret, frame = reader.read()

        if not ret:
            break

        # -------------------------
        # Detection + Tracking
        # -------------------------

        detections = tracker.track(frame)

        # -------------------------
        # Lane Detection
        # -------------------------

        frame = lane_detector.detect(frame)

        # -------------------------
        # FPS
        # -------------------------

        fps = fps_counter.update()

        # -------------------------
        # Draw detections
        # -------------------------

        frame = hud.draw_detections(
            frame,
            detections
        )

        # -------------------------
        # Draw trajectories
        # -------------------------

        
        # -------------------------
        # Draw FPS
        # -------------------------

        frame = hud.draw_fps(
            frame,
            fps
        )

        # -------------------------
        # Display
        # -------------------------

        cv2.imshow(
            "RoadWatch Copilot",
            frame
        )

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =========================
    # 5. Cleanup
    # =========================

    reader.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()