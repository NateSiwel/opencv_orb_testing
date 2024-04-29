import cv2
from numpy import who

class Display(object):
    def __init__(self):
        self.window_name = 'frame'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def draw_keypoints(self, frame, corners ):
        for i in corners:
            x,y = i.ravel()
            cv2.circle(frame,(x,y),3,255,-1)

        return frame 

    def close(self):
        cv2.destroyWindow(self.window_name)


