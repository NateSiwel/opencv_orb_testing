import cv2

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
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
            matches = self.bf.match(queryDescriptors=des, trainDescriptors=self.last['des'])

        self.last = {'kps':kps, 'des':des}
       
        return matches 
