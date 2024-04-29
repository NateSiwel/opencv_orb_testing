import cv2
from display import Display

vid = cv2.VideoCapture(-1)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

orb = cv2.ORB_create()

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def extract(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = orb.detect(gray, None)
        self.kp, self.des = orb.compute(gray, kp)

def main():
    display = Display()
    orb = FeatureExtractor()

    while True:
        ret, frame = vid.read()
        orb.extract(frame)
        frame = display.draw_keypoints(frame, orb.kp)

        display.show(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    display.close()

if __name__ == '__main__':
    main()
