import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import OrderedDict

#########################################
#constants
SPEED = 10 #pixel/s random speed for now
WIDTH_OF_CAR = 20 #pixels random width (back wheel to back wheel)
#########################################

#find coordinates of laser in a frame
def getLocation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
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
    '''
    mask = mask1 + mask2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #find contour from thresholded image:
    contours, heirarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(contours) > 0:
        print "found"
        largestContour = max(contours, key=cv2.contourArea)
         
        #calculate moments of inertia of shape - weighted average.
        moments = cv2.moments(largestContour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])) 

    return center

#transform coordinates into directions for each engine
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
                directions.append(curAngleTime) #tells angle to turn (if any)
            #get distance to move forward, if any
            curDistanceTime = math.hypot(vector2[0], vector2[1]) / SPEED
            if curDistanceTime != (0, 0):
                directions.append((curDistanceTime, curDistanceTime)) #investigate whether two motors at once gives different speed - must relate to rpm/SPEED issue
            orientation = vector2
            location = nextPoint #changed location
            #print "location: " + str(location) + " orientation: " + str(orientation)
            
    return directions        

#simplify curve so that all deviations smaller than epsilon are omitted, should help with jitterings in pixels. 
#need to find a way to remove consecutive equal points, but not ones that indicate visiting the same spot twice. possibly in perpendicularDistance
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
        # Recursive call
        recResults1 = douglasPeucker(pointList[:index + 1], epsilon)
        recResults2 = douglasPeucker(pointList[index:], epsilon)
 
        # Build the result list
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
#rectangle = list(OrderedDict.fromkeys(rectangle)) #introduces bug of never being able to go to the same exact spot twice.
#print len(douglasPeucker(rectangle, 1))
#print len(douglasPeucker(rectangle, 2))
#print len(douglasPeucker(rectangle, .5))
#print douglasPeucker(rectangle, 1)

#print bigJump
#print douglasPeucker(bigJump, 1)
'''
#########################################
#main body
laserArray = []
cap = cv2.VideoCapture("dark-box.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("dark-circle.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
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
        location = (location[0] * int(width / GOAL_WIDTH),  location[1] * int(height / GOAL_HEIGHT))
        #location[0] = location[0] * int(width / GOAL_WIDTH)
        #location[1] = location[1] * int(height / GOAL_HEIGHT)
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
plt.plot(xs, ys, 'o')
plt.show()
#print laserArray
'''
noDuplicates = list(OrderedDict.fromkeys(laserArray)) #introduces bug of never being able to go to the same exact spot twice.
simplifiedPath = douglasPeucker(noDuplicates, 1)
start = (int(width/2), 0) #starting position of the robot, might need to change.
orientation = (0, 1)
directions = pointsToDirections(simplifiedPath, start, orientation)
print directions'''

