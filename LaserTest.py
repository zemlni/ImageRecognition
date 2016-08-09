#import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#########################################
#constants
SPEED = 10 #pixel/s random speed for now
WIDTH_OF_CAR = 20 #pixels random width (back wheel to back wheel)
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

def pointsToDirections(laserArray, start):
    directions = [] #tuples of format (t1, t2) where t1 = time for engine 1(left) to run, t2 = time for engine 2(right) to run.

    #first maneuver: navigate to first point of path
    startVector = (start[0], start[1] + 1)#facing forward
    #print startVector
    point = laserArray[0]
    #print point
    pointVector = (point[0] - startVector[0], point[1] - startVector[1])
    #print pointVector
    directions.append(angleTime(startVector, pointVector)) #tells initial angle to turn (if any) 
    
    startDistanceTime = math.hypot(pointVector[0], pointVector[1]) / SPEED
    directions.append((startDistanceTime, startDistanceTime)) #investigate whether two motors at once gives different speed - must relate to rpm/SPEED issue
    location = point
    for i in range(1, len(laserArray) - 1):
        prevPoint = laserArray[i - 1]
        curPoint = laserArray[i]
        nextPoint = laserArray[i + 1]
	if not (location == nextPoint or location == prevPoint): 
            vector1 = (curPoint[0] - prevPoint[0], curPoint[1] - prevPoint[1])#need to compare location instead of using three consecutive points
            vector2 = (nextPoint[0] - curPoint[0], nextPoint[1] - curPoint[1])
            curAngleTime = angleTime(vector1, vector2)
            if curAngleTime != (0, 0):
                directions.append(curAngleTime) #tells angle to turn (if any)
                print "turn"

            curDistanceTime = math.hypot(vector2[0], vector2[1]) / SPEED
            directions.append((curDistanceTime, curDistanceTime)) #investigate whether two motors at once gives different speed - must relate to rpm/SPEED issue
            #print directions
            location = nextPoint #changed location
    return directions        

#transform angles into times for specific engine to run
def angleTime(v1, v2): 
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    bottom = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
    #print "v1: " + str(v1) + "v2: " + str(v2)
    angle = math.acos(dot / bottom)
    time = WIDTH_OF_CAR * angle / SPEED
    answer = (None, None)
    if (v1[0] * v2[1] - v1[1] * v2[0]) > 0: #2d-cross product tells you whether vector is to the right or left. negative = left, positive = right
    	answer = (0, time)
    else: answer = (0, time)
    return answer

def getAverage(pixel):
    average = 0
    for num in pixel:
            average += num
    return average / len(pixel[:3])
#########################################
#tests
#generate points for some shapes
n = 500 # number of points
r = 10
circle = [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)]
rectangle = []
for h1 in range(0, 10): rectangle.append((0, h1))
for l1 in range(0, 20): rectangle.append((l1, 9))
for h2 in range(9, -1, -1): rectangle.append((19, h2))
for l2 in range(19, -1, -1): rectangle.append((l2, 0))
start = (0, 0)
print rectangle
print pointsToDirections(rectangle, start)
#print pointsToDirections(circle, start)
'''
#########################################
laserArray = []
cap = cv2.VideoCapture("video2.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("video2.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"
width = cap.get(CV_CAP_PROP_FRAME_WIDTH)
height = cap.get(CV_CAP_PROP_FRAME_HEIGHT)
aspectRatio = width/height

while True:
    flag, frame = cap.read()
    if flag:
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        GOAL_WIDTH = 300 #change if you want different width, height will change accordingly.
        GOAL_HEIGHT = GOAL_WIDTH / aspectRatio
        size = (GOAL_WIDTH, int(GOAL_HEIGHT)) #might be the other way around, need to check
        frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA);
        location =  getLocation(frame)
	#print "frame number" + str(pos_frame) 
        #xs = [x[0] for x in test]
        #ys = [x[1] for x in test]
        #plt.plot(xs, ys, 'o')
        #plt.show()
        #need to magnify locations by aspect ratio to reflect original video, this should do it, need to check.
        location[0] = location[0] * int(width / GOAL_WIDTH)
        location[1] = location[1] * int(height / GOAL_HEIGHT)
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

start = (int(width/2), 0) #starting position of the robot, might need to change.
directions = pointsToDirections(laserArray, start)
'''
