import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexcity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexcity = complexcity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexcity, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for hand_Lms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_Lms, self.mpHands.HAND_CONNECTIONS)
        return img

def main():
    pTime = cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)

        cv2.imshow("Image", img)
        cv2.waitKey(2)

    
print(__name__,"__name__")
if __name__ == "__main__":
    main()