# FrameManager.py

# External imports
import cv2
import numpy as np

class FrameManager:
    """
    This class helps with modifying thing in a frame
    """
    def __init__(self, color_table=None, unknown_color=None):
        self.color_table = color_table  # A table to look up color indexes in
        self.unknown_color = unknown_color

    # Draw a line
    def draw_line(self, frame, p0, p1, color, thickness):
        frame = cv2.line(frame, p0, p1, color, thickness)
        return frame

    # Place a rect
    def place_rect(self, frame, x, y, w, h, color):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        return frame

    # Place several rect
    def place_rects(self, frame, coords, color):
        for coord in coords:
            self.place_rect(frame, coord[0], coord[1], coord[2], coord[3], color)
        return frame

    @staticmethod
    # Resize a frame
    def resize_frame(frame, res):
        return cv2.resize(frame, res)

    # Label a cone according to its coordinate and color index
    # This places a colored rect over the cone and prints in color
    def label_cone(self, frame, coord, color_idx):
        if isinstance(color_idx, int):
            color = self.color_table[color_idx]
        else:
            color = self.unknown_color

        self.place_rect(frame=frame,
                        x=coord[0],
                        y=coord[1],
                        w=coord[2],
                        h=coord[3],
                        color=color[1])
        cv2.putText(frame,
                    color[0],
                    (int(coord[0]), int(coord[1]) - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 0), 2, cv2.LINE_AA)

    # Label multiple cones
    def label_cones(self, frame, cones):
        for cone in cones:
            self.label_cone(frame=frame, coord=cone.cone_window, color_idx=cone.color)
        return frame

    # Make a gray copy of the frame
    def create_gray_frame(self, frame):
        return cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2GRAY)

    # Make a hsv copy of the frame
    def create_hsv_frame(self, frame):
        return cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2HSV)

    # Make a gaussian filtered copy of the frame
    def create_gaussian_frame(self, frame, kernel):
        return cv2.GaussianBlur(np.copy(frame), kernel, 0)
