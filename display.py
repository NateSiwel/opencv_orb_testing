import cv2
from numpy import who

class Display(object):
    def __init__(self):
        self.window_name = 'frame'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def draw_keypoints(self, frame, kps ):
        frame = cv2.drawKeypoints(frame, kps, 0, color=(255,0,255), flags=0)

        return frame 

    def close(self):
        cv2.destroyWindow(self.window_name)


