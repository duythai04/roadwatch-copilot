import cv2


class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

    def read(self):
        return self.cap.read()

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        self.cap.release()