import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)
startDist = None
scale = 0
cx, cy = 500,500
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread("cvarduino.jpg")

    if len(hands) == 2:
        # print(detector.fingersUp(hands&#91;0]), detector.fingersUp(hands&#91;1]))
        if detector.fingersUp(hands&#91;0]) == &#91;1, 1, 0, 0, 0] and \
                detector.fingersUp(hands&#91;1]) == &#91;1, 1, 0, 0, 0]:
            # print("Zoom Gesture")
            lmList1 = hands&#91;0]&#91;"lmList"]
            lmList2 = hands&#91;1]&#91;"lmList"]
            # point 8 is the tip of the index finger
            if startDist is None:
                #length, info, img = detector.findDistance(lmList1&#91;8], lmList2&#91;8], img)
                length, info, img = detector.findDistance(hands&#91;0]&#91;"center"], hands&#91;1]&#91;"center"], img)

                startDist = length

            #length, info, img = detector.findDistance(lmList1&#91;8], lmList2&#91;8], img)
            length, info, img = detector.findDistance(hands&#91;0]&#91;"center"], hands&#91;1]&#91;"center"], img)

            scale = int((length - startDist) // 2)
            cx, cy = info&#91;4:]
            print(scale)
    else:
        startDist = None

    try:
        h1, w1, _= img1.shape
        newH, newW = ((h1+scale)//2)*2, ((w1+scale)//2)*2
        img1 = cv2.resize(img1, (newW,newH))

        img&#91;cy-newH//2:cy+ newH//2, cx-newW//2:cx+ newW//2] = img1
    except:
        pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)
