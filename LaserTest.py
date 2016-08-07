import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    for row in range(0, len(imageArray)):
            for col in range(0, len(imageArray[row])):
                pixel = imageArray[row][col]
                testAverage = getAverage(pixel)
                if testAverage > balance:
                    light.append((row, col))
		    #print str(row) + ", " + str(col) + "light" 
                else:
                    dark.append((row, col))
		    #print "dark"
    #assumption that laser points take up less space in the picture than non laser.
    if len(light) > len(dark):
        return dark
    else: return light

def getLocation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    '''
    #red can have two h values 0 and 180, check both
    #0:
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #180:
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    '''
    #white
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    #mask = mask1 + mask2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #find contour from thresholded image:
    contours, heirarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
         
        #calculate moments of inertia of shape - weighted average.
        moments = cv2.moments(largestContour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])) 

    return center

def getAverage(pixel):
    average = 0
    for num in pixel:
            average += num
    return average / len(pixel[:3])

#########################################
laserArray = []
cap = cv2.VideoCapture("video2.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("video2.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"



while True:
    flag, frame = cap.read()
    if flag:
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        ratio = 300.0 / frame.shape[1]
        height = (300, int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, height,  interpolation = cv2.INTER_AREA);
        location =  getLocation(frame)
	#print "frame number" + str(pos_frame) 
        #xs = [x[0] for x in test]
        #ys = [x[1] for x in test]
        #plt.plot(xs, ys, 'o')
        #plt.show()
        laserArray.append(location)        
        
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
xs = [x[0] for x in laserArray]
ys = [x[1] for x in laserArray]
plt.plot(xs, ys, '-')
plt.show()
print laserArray
