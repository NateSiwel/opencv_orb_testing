import cv2

class Display(object):
    def __init__(self):
        self.window_name = 'frame'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def draw_keypoints(self, frame, keypoints):
        return cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

    def close(self):
        cv2.destroyWindow(self.window_name)


