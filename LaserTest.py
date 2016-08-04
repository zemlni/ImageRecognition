import cv2
import numpy as np

#########################################

#change this so that it doesn't have to pass through the whole array three times.
def getLaserPoints(imageArray):
#        balanceArray = []
#        for row in imageArray:
#                for pixel in row:
#                        average = getAverage(pixel)
#                        balanceArray.append(average)
#        balance = 0
#        for num in balanceArray:
#                balance += num
#        balance = balance / len(balanceArray)
    balance = np.mean(imageArray)
    light = []
    dark = []
    for row in len(answer):
            for col in len(imageArray[row]):
                pixel = answer[row][col]
                testAverage = getAverage(pixel)
                if testAverage > balance:
                    light.append((row, col))
                else:
                    dark.append((row, col))
    #assumption that laser points take up less space in the picture than non laser.
    if len(light) > len(dark):
        return dark
    else: return light

def getAverage(pixel):
    average = 0
    for num in pixel:
            average += num
    return average / len(pixel[:3])

#########################################
laserArray = []
cap = cv2.VideoCapture("video1.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("video1.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"

pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        print str(pos_frame)+" frames"
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print "frame is not ready"
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
    if frame != None:
        imageArray = np.array(frame)
        laserArray += getLaserPoints(imageArray)
print laserArray
#comment 
