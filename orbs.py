import cv2
from display import Display
from featureExtractor import FeatureExtractor 
import numpy as np

#vid = cv2.VideoCapture(-1)
vid = cv2.VideoCapture('driving.mp4')
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

W = 1920//2
H = 1080//2

F = 1
display = Display(W, H)

#camera matrix
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])


orb = FeatureExtractor(K)

def process_frame(frame):
    matches = orb.extract(frame)
    if matches is None:
        return frame 
    last = orb.last 
    kps = last['kps']
    des = last['des']
    
    if kps is not None:
        frame = display.draw_keypoints(frame, kps, matches, orb.denormalize)

    return frame

def main():
    while True:
        ret, frame = vid.read()

        frame = process_frame(frame)
        
        display.show(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    display.close()

if __name__ == '__main__':
    main()
