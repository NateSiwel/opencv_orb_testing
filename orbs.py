import cv2
from display import Display
from featureExtractor import FeatureExtractor 

vid = cv2.VideoCapture(-1)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

display = Display()
orb = FeatureExtractor()

def process_frame(frame):
    matches = orb.extract(frame)
    if matches is None:
        return frame 
    last = orb.last 
    kps = last['kps']
    des = last['des']
    
    if kps is not None:
        frame = display.draw_keypoints(frame, kps, matches)

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
