import cv2
import numpy as np


class LaneDetector:

    def __init__(self):
        # Lưu lane hiện tại
        self.left_lane = None
        self.right_lane = None

    def detect(self, frame):
        """
        Phát hiện lane trái và lane phải.

        Returns:
            lane_frame: frame đã được vẽ lane
        """

        height, width = frame.shape[:2]

        # ==================================================
        # 1. Chuyển sang HLS để phát hiện màu trắng / vàng
        # ==================================================

        hls = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HLS
        )

        # ==================================================
        # 2. White lane mask
        # ==================================================

        lower_white = np.array(
            [0, 140, 0],
            dtype=np.uint8
        )

        upper_white = np.array(
            [255, 255, 255],
            dtype=np.uint8
        )

        white_mask = cv2.inRange(
            hls,
            lower_white,
            upper_white
        )

        # ==================================================
        # 3. Yellow lane mask
        # ==================================================

        lower_yellow = np.array(
            [10, 30, 70],
            dtype=np.uint8
        )

        upper_yellow = np.array(
            [40, 220, 255],
            dtype=np.uint8
        )

        yellow_mask = cv2.inRange(
            hls,
            lower_yellow,
            upper_yellow
        )

        # Gộp trắng + vàng
        color_mask = cv2.bitwise_or(
            white_mask,
            yellow_mask
        )

        # ==================================================
        # 4. Grayscale
        # ==================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # 5. Gaussian Blur
        # ==================================================

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ==================================================
        # 6. Canny Edge Detection
        # ==================================================

        edges = cv2.Canny(
            blur,
            30,
            100
        )

        # ==================================================
        # 7. Kết hợp Canny + Color
        # ==================================================
        #
        # QUAN TRỌNG:
        #
        # Trước đây dùng bitwise_and()
        # khiến điều kiện quá chặt.
        #
        # Bây giờ dùng OR:
        #
        # edge HOẶC pixel có màu lane
        #

        combined = cv2.bitwise_or(
            edges,
            color_mask
        )

        # ==================================================
        # 8. Region Of Interest
        # ==================================================

        mask = np.zeros_like(
            combined
        )

        polygon = np.array(
            [[
                # Bottom-left
                (
                    int(width * 0.15),
                    int(height * 0.78)
                ),

                # Top-left
                (
                    int(width * 0.42),
                    int(height * 0.52)
                ),

                # Top-right
                (
                    int(width * 0.58),
                    int(height * 0.52)
                ),

                # Bottom-right
                (
                    int(width * 0.85),
                    int(height * 0.78)
                )
            ]],
            dtype=np.int32
        )

        cv2.fillPoly(
            mask,
            polygon,
            255
        )

        # Chỉ giữ phần nằm trong ROI
        cropped_edges = cv2.bitwise_and(
            combined,
            mask
        )

        # ==================================================
        # 9. Debug cửa sổ Edge
        # ==================================================

        cv2.imshow(
            "Lane Debug - Edges",
            cropped_edges
        )

        # ==================================================
        # 10. Hough Line Transform
        # ==================================================

        lines = cv2.HoughLinesP(
            cropped_edges,

            rho=1,

            theta=np.pi / 180,

            # Giảm threshold để dễ phát hiện hơn
            threshold=10,

            # Chấp nhận line ngắn hơn
            minLineLength=10,

            # Cho phép nối các đoạn đứt
            maxLineGap=100
        )

        # ==================================================
        # 11. Frame dùng để hiển thị
        # ==================================================

        lane_frame = frame.copy()

        # Vẽ ROI màu xanh để debug
        cv2.polylines(
            lane_frame,
            polygon,
            True,
            (255, 0, 0),
            2
        )

        # ==================================================
        # 12. Danh sách lane candidate
        # ==================================================

        left_fit = []
        right_fit = []

        hough_count = 0

        # ==================================================
        # 13. Xử lý Hough lines
        # ==================================================

        if lines is not None:

            # Chuẩn hóa về:
            #
            # [
            #   [x1,y1,x2,y2],
            #   [x1,y1,x2,y2],
            #   ...
            # ]

            lines = np.asarray(
                lines
            ).reshape(-1, 4)

            hough_count = len(lines)

            for x1, y1, x2, y2 in lines:

                # ------------------------------------------
                # Debug: vẽ Hough candidate màu vàng
                # ------------------------------------------

                cv2.line(
                    lane_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                # ------------------------------------------
                # Tránh chia cho 0
                # ------------------------------------------

                if x2 == x1:
                    continue

                # ------------------------------------------
                # Tính slope
                # ------------------------------------------

                slope = (
                    (y2 - y1)
                    /
                    (x2 - x1)
                )

                # ------------------------------------------
                # Loại line gần nằm ngang
                # ------------------------------------------

                if abs(slope) < 0.25:
                    continue

                # ------------------------------------------
                # Loại line quá dốc
                # ------------------------------------------

                if abs(slope) > 5.0:
                    continue

                # ------------------------------------------
                # Tính intercept
                #
                # y = slope*x + intercept
                # ------------------------------------------

                intercept = (
                    y1
                    -
                    slope * x1
                )

                # Tâm của đoạn line
                center_x = (
                    x1 + x2
                ) / 2

                # ------------------------------------------
                # LEFT LANE
                # ------------------------------------------

                if (
                    slope < 0
                    and center_x < width / 2
                ):

                    left_fit.append(
                        (
                            slope,
                            intercept
                        )
                    )

                # ------------------------------------------
                # RIGHT LANE
                # ------------------------------------------

                elif (
                    slope > 0
                    and center_x > width / 2
                ):

                    right_fit.append(
                        (
                            slope,
                            intercept
                        )
                    )

        # ==================================================
        # 14. Tính lane trung bình
        # ==================================================

        left_lane = None
        right_lane = None

        # -------------------------
        # LEFT
        # -------------------------

        if len(left_fit) > 0:

            left_average = np.mean(
                left_fit,
                axis=0
            )

            left_lane = self.make_coordinates(
                frame,
                left_average
            )

        # -------------------------
        # RIGHT
        # -------------------------

        if len(right_fit) > 0:

            right_average = np.mean(
                right_fit,
                axis=0
            )

            right_lane = self.make_coordinates(
                frame,
                right_average
            )

        # Lưu lại
        self.left_lane = left_lane
        self.right_lane = right_lane

        # ==================================================
        # 15. Vẽ LEFT lane
        # ==================================================

        if left_lane is not None:

            cv2.line(
                lane_frame,

                (
                    left_lane[0],
                    left_lane[1]
                ),

                (
                    left_lane[2],
                    left_lane[3]
                ),

                # RED
                (0, 0, 255),

                6
            )

        # ==================================================
        # 16. Vẽ RIGHT lane
        # ==================================================

        if right_lane is not None:

            cv2.line(
                lane_frame,

                (
                    right_lane[0],
                    right_lane[1]
                ),

                (
                    right_lane[2],
                    right_lane[3]
                ),

                # RED
                (0, 0, 255),

                6
            )

        # ==================================================
        # 17. Debug information
        # ==================================================

        cv2.putText(
            lane_frame,

            f"Hough: {hough_count}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2
        )

        cv2.putText(
            lane_frame,

            f"Left: {len(left_fit)} Right: {len(right_fit)}",

            (20, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2
        )

        return lane_frame

    # ======================================================
    # Convert slope/intercept → coordinates
    # ======================================================

    def make_coordinates(
        self,
        frame,
        line_parameters
    ):

        slope, intercept = line_parameters

        height, width = frame.shape[:2]

        # Tránh slope = 0
        if abs(slope) < 0.001:
            return None

        # =========================
        # Điểm dưới của lane
        # =========================

        y1 = int(
            height * 0.78
        )

        # =========================
        # Điểm trên của lane
        # =========================

        y2 = int(
            height * 0.55
        )

        # =========================
        # Tính X
        # =========================

        x1 = int(
            (y1 - intercept)
            /
            slope
        )

        x2 = int(
            (y2 - intercept)
            /
            slope
        )

        # =========================
        # Giới hạn tọa độ
        # =========================

        x1 = np.clip(
            x1,
            0,
            width - 1
        )

        x2 = np.clip(
            x2,
            0,
            width - 1
        )

        return (
            int(x1),
            y1,
            int(x2),
            y2
        )

    # ======================================================
    # Getter
    # ======================================================

    def get_lanes(self):
        """
        Trả về tọa độ lane hiện tại.

        Sau này LaneDepartureDetector
        sẽ sử dụng dữ liệu này.
        """

        return (
            self.left_lane,
            self.right_lane
        )