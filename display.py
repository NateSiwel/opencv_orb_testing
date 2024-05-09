import cv2
import numpy as np

class Display(object):
    def __init__(self, W, H):
        self.window_name = 'frame'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, W, H)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def draw_keypoints(self, frame, kps, matches, denormalize):
        #cv2.drawKeypoints(frame, kps, frame, color=(255,0,255), flags=0)
        current_coords = [match[0] for match in matches]
        old_coords = [match[1] for match in matches]

        for i in range(len(current_coords)):
            current_point = denormalize(current_coords[i])
            old_point = denormalize(old_coords[i])

            current_x = int(current_point[0])
            current_y = int(current_point[1])
            old_x = int(old_point[0])
            old_y = int(old_point[1])

            cv2.line(frame, (current_x, current_y), (old_x, old_y), color=(0, 255, 0), thickness=2)

            #cv2.drawKeypoints(frame, kps, frame, color=(255,255,255), flags=0)

        return frame 

    def close(self):
        cv2.destroyWindow(self.window_name)


