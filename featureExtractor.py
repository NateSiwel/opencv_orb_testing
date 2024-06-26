import cv2
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform

def add_ones(x):
      return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class FeatureExtractor:
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(K)


    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))
    
    def extract(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detect
        corners = cv2.goodFeaturesToTrack(gray, 3000, qualityLevel=0.01, minDistance=8)

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


            matches[:, 0, :] = np.dot(self.Kinv,add_ones(matches[:, 0, :]).T).T[:, 0:2]
            matches[:, 1, :] = np.dot(self.Kinv,add_ones(matches[:, 1, :]).T).T[:, 0:2]

            try: 
                #work in progress
                model, inliers = ransac(
                    (matches[:, 0],matches[:, 1]),
                    #EssentialMatrixTransform,
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
