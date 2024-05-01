import cv2
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detect
        corners = cv2.goodFeaturesToTrack(gray, 3000, qualityLevel=0.01, minDistance=10)

        #extract
        kps = []
        for i in corners:
            x,y = i.ravel()
            kps.append(cv2.KeyPoint(x, y, 0))
        kps,des = self.orb.compute(gray, kps)

        #match
        matches = None
        if (self.last is not None):
            matches = self.bf.knnMatch(des,self.last['des'],k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.65*n.distance:
                    good.append((kps[m.queryIdx].pt, self.last['kps'][m.trainIdx].pt))

            matches = good
            matches = np.array(matches)
            
            try: 
                #work in progress
                model, inliers = ransac(
                    (matches[:, 0],matches[:, 1]),
                    FundamentalMatrixTransform,
                    min_samples=8,
                    residual_threshold=1,
                    max_trials=100,
                )

                matches = matches[inliers]

            except ValueError as e:
                print(f"error: {str(e)}")
        self.last = {'kps':kps, 'des':des}
       
        return matches 
