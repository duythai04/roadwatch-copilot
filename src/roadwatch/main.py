import cv2

from roadwatch.core.video_reader import VideoReader
from roadwatch.tracking.vehicle_tracker import VehicleTracker
from roadwatch.visualization.hud import HUD
from roadwatch.utils.fps import FPSCounter


def main():

    video_path = "data/raw/videos/test.mp4"

    # Video reader
    reader = VideoReader(video_path)

    # Tracker
    tracker = VehicleTracker(
        model_path="yolo11n.pt",
        confidence=0.4
    )

    # hub
    hud = HUD()

    # FPS counter
    fps_counter = FPSCounter()

    print("Video FPS:", reader.get_fps())
    print("Width:", reader.get_width())
    print("Height:", reader.get_height())

    while True:



        # 1. read frame

        ret, frame = reader.read()

        if not ret:
            break

      
        # 2. detection + tracking

        detections = tracker.track(frame)

     

        # 3. fps

        fps = fps_counter.update()

    
        # 4. draw bounding boxes
  

        frame = hud.draw_detections(
            frame,
            detections
        )

        



        # 5. draw fps

        frame = hud.draw_fps(
            frame,
            fps
        )

    

        # 6. display

        cv2.imshow(
            "RoadWatch Copilot - Tracking",
            frame
        )

        # q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()