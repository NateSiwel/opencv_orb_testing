import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def extract(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 3000, qualityLevel=0.01, minDistance=10)
        corners = np.intp(corners)
        self.corners = corners


