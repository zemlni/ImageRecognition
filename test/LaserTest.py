import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import RPi.GPIO as GPIO
from datetime import datetime

#########################################
#constants
SPEED = 10 #pixel/s random speed for now
WIDTH_OF_CAR = 20 #pixels random width (back wheel to back wheel)
HUE_MIN = 20
HUE_MAX = 160
SAT_MIN = 100
SAT_MAX = 255
VAL_MIN = 200
VAL_MAX = 255
channels = {
    'hue': None,
    'saturation': None,
    'value': None,
    'laser': None,
}
#pins for left and right engines.
LEFT_PIN = 
RIGHT_PIN = 
#########################################

#read video from file
def readFromFile(path):
    laserArray = []
    cap = cv2.VideoCapture("light-circle.mp4")
    while not cap.isOpened():
        cap = cv2.VideoCapture("light-circle.mp4")
        cv2.waitKey(1000)

    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    aspectRatio = width/height

    while True:
        flag, frame = cap.read()
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            #need to check whether you are ever allowed to look at the last frame or not
            break
        if flag:
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            GOAL_WIDTH = 300 #change if you want different width, height will change accordingly.
            GOAL_HEIGHT = GOAL_WIDTH / aspectRatio
            size = (GOAL_WIDTH, int(GOAL_HEIGHT)) #might be the other way around, need to check
            frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA);
            location =  getLocation(frame)
            if location == None: continue

            #magnify locations to match original video size, this should do it, need to check.
            location = (location[0] * int(width / GOAL_WIDTH),  location[1] * int(height / GOAL_HEIGHT))
            if (len(laserArray) > 0 and laserArray[-1] != location) or (len(laserArray) == 0):
                laserArray.append(location)

        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print "frame is not ready"
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
    return laserArray

#wrapper for cv2.threshold function.
def threshold_image(channel):
    if channel == "hue":
        minimum = HUE_MIN
        maximum = HUE_MAX
    elif channel == "saturation":
        minimum = SAT_MIN
        maximum = SAT_MAX
    elif channel == "value":
        minimum = VAL_MIN
        maximum = VAL_MAX

    (t, tmp) = cv2.threshold(
        channels[channel], # src
        maximum, # threshold value
        0, # we dont care because of the selected type
        cv2.THRESH_TOZERO_INV # type
    )

    (t, channels[channel]) = cv2.threshold(
        tmp, # src
        minimum, # threshold value
        255, # maxvalue
        cv2.THRESH_BINARY # type
    )

    if channel == 'hue':
        # only works for filtering red color because the range for the hue is split
        channels['hue'] = cv2.bitwise_not(channels['hue'])

#get location of the laser given a frame
def getLocation(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # split the video frame into color channels
    h, s, v = cv2.split(hsv_img)
    channels['hue'] = h
    channels['saturation'] = s
    channels['value'] = v

    # Threshold ranges of HSV components; storing the results in place
    threshold_image("hue")
    threshold_image("saturation")
    threshold_image("value")

    # Perform an AND on HSV components to identify the laser!
    channels['laser'] = cv2.bitwise_and(channels['hue'], channels['value'])

    channels['laser'] = cv2.bitwise_and(channels['saturation'], channels['laser'])

    #track
    mask = channels['laser']
    center = None
    countours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(countours) > 0:
        #print "found"
        c = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            center = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        else:
            center = int(x), int(y)
    #print center
    return center

#transform coordinates into directions for each engine and execute them
def pointsToDirections(laserArray, startLocation, startOrientation):
    directions = [] #tuples of format (t1, t2) where t1 = time for engine 1(left) to run, t2 = time for engine 2(right) to run.
    
    location = startLocation
    orientation = startOrientation
    for nextPoint in laserArray:
	if not (location == nextPoint): #checks for zero vectors
            vector1 = orientation 
            vector2 = (nextPoint[0] - location[0], nextPoint[1] - location[1])
            #get angle to turn, if any
            curAngleTime = angleTime(vector1, vector2)
            if curAngleTime != (0, 0):
                executeDirections(curAngleTime)
                directions.append(curAngleTime)
            #get distance to move forward, if any
            curDistanceTime = math.hypot(vector2[0], vector2[1]) / SPEED
            if curDistanceTime != 0:
                executeDirections((curDistanceTime, curDistanceTime))
                directions.append((curDistanceTime, curDistanceTime)) #investigate whether two motors at once gives different speed - must relate to rpm/SPEED issue
            orientation = vector2
            location = nextPoint #changed location
            
    return directions        

#execute directions given by function pointsToDirections.
def executeDirections(direction):
    l = []
    if direction[0] != 0:
        l.append(LEFT_PIN)
    if direction[1] != 0:
        l.append(RIGHT_PIN)
    GPIO.output(l, 1)
    startTime = datetime.now()
    while getDifference(startTime, dateTime.now()) > l[0]: continue
    GPIO.output(l, 0)

#return difference between two datetime objects.
def getDifference(time1, time2):
    return ((time2.days * 24 * 60 * 60 + time2.seconds) * 1000 + time2.microseconds / 1000.0) - ((time1.days * 24 * 60 * 60 + time1.seconds) * 1000 + time1.microseconds / 1000.0)

#simplify curve so that all deviations smaller than epsilon are omitted, should help with jitterings in pixels. 
def douglasPeucker(pointList, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(pointList) - 1
    for i in range(1, end):
        d = perpendicularDistance(pointList[i], (pointList[0], pointList[end])) 
        if d > dmax: 
            index = i
            dmax = d
    
    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        recResults1 = douglasPeucker(pointList[:index + 1], epsilon)
        recResults2 = douglasPeucker(pointList[index:], epsilon)
 
        resultList = recResults1[:-1] + recResults2 #to avoid having index twice.
    else:
        resultList = [pointList[0], pointList[end]]
    
    return resultList

#get perpendicular distance between a point and a line
def perpendicularDistance(point, line): 
    #a = h*b/2 then h = 2a/b
    (l1, l2) = line
    a2 = math.fabs((l2[1] - l1[1]) * point[0] - (l2[0] - l1[0]) * point[1] + l2[0] * l1[1] - l2[1] * l1[0])
    b = math.hypot(l2[0] - l1[0], l2[1] - l1[1])
    return a2 / b

#transform angles into times for specific engine to run
def angleTime(v1, v2): 
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    bottom = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
    angle = math.acos(dot / bottom)
    time = WIDTH_OF_CAR * angle / SPEED
    answer = (None, None)
    if (v1[0] * v2[1] - v1[1] * v2[0]) > 0: #2d-cross product tells you whether vector is to the right or left. negative = left, positive = right
    	answer = (0, time)
    else: answer = (time, 0)
    return answer

def getAverage(pixel):
    average = 0
    for num in pixel:
            average += num
    return average / len(pixel[:3])

#########################################
#tests
#generate points for some shapes
'''
n = 500 # number of points
r = 100
circle = [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)]
rectangle = []
for h1 in range(0, 10): rectangle.append((0, h1))
for l1 in range(0, 20): rectangle.append((l1, 9))
for h2 in range(9, -1, -1): rectangle.append((19, h2))
for l2 in range(19, -1, -1): rectangle.append((l2, 0))
fractions = [(.5, .5), (1.5, 1.5)]
bigJump = [(0, 0), (100, 0), (0, 0)]
sameSpot = [(1, 1) for x in range(0, 10)]
start = (0, 0)
orientation = (0, 1)
'''

### tests for pointsToDirections ###
#print rectangle 
#print pointsToDirections(rectangle, start, orientation)

#print circle 
#print pointsToDirections(circle, start, orientation)

#print sameSpot 
#print pointsToDirections(sameSpot, start, orientation)

#print bigJump
#print pointsToDirections(bigJump, start, orientation)

#print fractions
#print pointsToDirections(fractions, start, orientation)
'''
### tests for perpendicularDistance ###
#p1 = (0, 1)
#line1 = ((-1, 0), (1, 0))
#print perpendicularDistance(p1, line1)

#p2 = (1, 0)
#line2 = ((0, 1), (0, -1))
#print perpendicularDistance(p2, line2)
#print perpendicularDistance(p2, line1)  

#p3 = (0, 0)
#line3 = ((0, 1), (1, 0))
#print perpendicularDistance(p3, line3)

### tests for douglasPeucker ###
#simpleCircle1 = douglasPeucker(circle, 1)
#print len(simpleCircle1)
#print simpleCircle1
#xs = [x[0] for x in simpleCircle1]
#ys = [x[1] for x in simpleCircle1]
#plt.plot(xs, ys, '--')
#xc = [x[0] for x in circle]
#yc = [x[1] for x in circle]
#plt.plot(xc, yc, '-')
#plt.show()

#print len(douglasPeucker(circle, .5))
#print len(douglasPeucker(circle, .1))
#print len(douglasPeucker(circle, 2))
#print len(douglasPeucker(circle, 5))
#make more tests to test out reduction of pixel jittering 

#print len(rectangle)
#print len(douglasPeucker(rectangle, 1))
#print len(douglasPeucker(rectangle, 2))
#print len(douglasPeucker(rectangle, .5))
#print douglasPeucker(rectangle, 1)

#print bigJump
#print douglasPeucker(bigJump, 1)
'''
#########################################
#main body

GPIO.setmode(GPIO.BOARD)
GPIO.setup([LEFT_PIN, RIGHT_PIN], GPIO.OUT)
laserArray = readFromFile("light-circle.mp4")

#simplifiedPath = douglasPeucker(laserArray, 10)
xs = [x[0] for x in laserArray]
ys = [x[1] for x in laserArray]
plt.plot(xs, ys, '-')
plt.show()
#print len(simplifiedPath)
print len(laserArray)
'''start = (int(width/2), 0) #starting position of the robot, might need to change.
orientation = (0, 1)
directions = pointsToDirections(simplifiedPath, start, orientation)
print directions'''
