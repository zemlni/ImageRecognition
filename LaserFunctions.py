import cv2
import numpy as np
#import matplotlib.pyplot as plt
import math
import RPi.GPIO as GPIO
from datetime import datetime
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
from multiprocessing import Queue, Lock
#from Queue import Empty
from collections import deque

#########################################
SPEED = 600  # pixel/s random speed for now
WIDTH_OF_CAR = 200  # pixels random width (back wheel to back wheel)
HUE_MIN = 10
HUE_MAX = 170
SAT_MIN = 100
SAT_MAX = 255
VAL_MIN = 100
VAL_MAX = 255
channels = {
    'hue': None,
    'saturation': None,
    'value': None,
    'laser': None,
}
# pins for left and right engines.
Side1Pin1   = 11    # pin11
Side1Pin2   = 12    # pin12
Side1Enable = 13    # pin13

Side2Pin1   = 15
Side2Pin2   = 16
Side2Enable = 18
#########################################
def extractLocation(queue, lock):
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.vflip = True
    #camera.contrast = 20
    camera.led = False
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)
    i = 0
    # capture frames from the camera
    previous = None
    M = getPerspectiveTransform()
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        #for frame in camera.capture_continuous(rawCapture, use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        #cv2.imwrite('image' + str(i) + '.png',image)

        # show the frame
        #cv2.imshow("Frame", image)
        #cv2.imwrite('image' + str(i) + '.png',image)
        #key = cv2.waitKey(1) & 0xFF
        #cv2.imwrite('image' + str(i) + '.png',image)
        image = cv2.warpPerspective(image, M, (640, 480))
        #cv2.imwrite('warpedimage' + str(i) + '.png',image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        location = getLocation(image)
        print i, location
        if location is not None: # and (previous is None or math.hypot(location[0] - previous[0], location[1] - previous[1]) > 10) and location is not None: #WRONG - previous doesn't represent anything
            queue.put(location)
            #previous = location
            cv2.circle(image, (location), 5, (0,255,0), 3)
        #yield location
        cv2.imwrite('image' + str(i) + '.png',image)
        i += 1

def move(queue, lock):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([Side1Pin1, Side1Pin2, Side1Enable, Side2Pin1, Side2Pin2, Side2Enable], GPIO.OUT)
    directions = []
    while True:
        print "TEST"
        lock.acquire()
        nextLocation = queue.get()
        lock.release()
        '''
        if len(queue) <= 0: 
            print "EMPTY"
            continue
        '''
        #nextLocation = queue.pop()
        print "move", nextLocation
        vector1 = (0, -1)
        vector2 = (nextLocation[0] - 640/2, nextLocation[1] - 480)
        curAngleTime, angle = angleTime(vector1, vector2)
        if curAngleTime != (0, 0):
            executeDirections(curAngleTime)
            directions.append(curAngleTime) #debugging purposes
        #queue = rotate(queue, angle)
        lock.acquire()
        rotate(queue, angle)
        lock.release()
        curDistanceTime = math.hypot(vector2[0], vector2[1])/SPEED
        if curDistanceTime != 0:
            executeDirections((curDistanceTime, curDistanceTime))
            directions.append((curDistanceTime, curDistanceTime))
        lock.acquire()
        translate(queue, vector2)
        lock.release()
        #time.sleep(.01)

def rotate(queue, angle):
    answer = Queue()
    matrix = np.matrix([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    print "rotate"
    
    while not queue.empty():
        elem = queue.get()
        elem = np.matrix(elem).transpose()
        mul = (matrix * elem).A.T
        elem = tuple([i for x in mul for i in x])
        answer.put(elem)
    #queue = answer
    while not answer.empty():
        queue.put(answer.get())
    '''
    for i in range(0, len(queue)):
        elem = queue[i]
        elem = np.matrix(elem).transpose()
        mul = (matrix * elem).A.T
        elem = tuple([i for x in mul for i in x])
        queue[i] = elem 
    '''

def translate(queue, newLocation):
    answer = Queue()
    print "translate"
    
    while not queue.empty():
        elem = queue.get()
        elem = (newLocation[0] - elem[0], newLocation[1] - elem[1])
        answer.put(elem)
    while not answer.empty():
        queue.put(answer.get())
    #queue = answer
    '''
    for i in range(0, len(queue)):
        elem = queue[i]
        queue[i] = (newLocation[0] - elem[0], newLocation[1] - elem[1])
    '''

def getPerspectiveTransform():
    pts = np.array([(628, 326), (447, 280), (152, 291), (44, 344)], dtype="float32")
    #dst = np.array([(640 - 172, 480 - 126), (640 - 172, 126), (172, 126), (172, 480 - 126)], dtype="float32")
    dst = np.array([(640 - round((640 - 147.5) / 2), 340), (640 - round((640 - 147.5) / 2), 340 - 114), (round((640 - 147.5) / 2), 340 - 114), (round((640 - 147.5) / 2), 340)], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return M


# read video from file
def readFromFile(path):
    laserArray = []
    cap = cv2.VideoCapture(path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)

    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    aspectRatio = width / height

    while True:
        flag, frame = cap.read()
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            # need to check whether you are ever allowed to look at the last frame or not
            break
        if flag:
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            GOAL_WIDTH = 300  # change if you want different width, height will change accordingly.
            GOAL_HEIGHT = GOAL_WIDTH / aspectRatio
            size = (GOAL_WIDTH, int(GOAL_HEIGHT))  # might be the other way around, need to check
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            location = getLocation(frame)
            if location == None: continue

            # magnify locations to match original video size, this should do it, need to check.
            location = (location[0] * int(width / GOAL_WIDTH), location[1] * int(height / GOAL_HEIGHT))
            if (len(laserArray) > 0 and laserArray[-1] != location) or (len(laserArray) == 0):
                laserArray.append(location)

        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame - 1)
            print
            "frame is not ready"
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
    return laserArray


# wrapper for cv2.threshold function.
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
        channels[channel],  # src
        maximum,  # threshold value
        0,  # we dont care because of the selected type
        cv2.THRESH_TOZERO_INV  # type
    )

    (t, channels[channel]) = cv2.threshold(
        tmp,  # src
        minimum,  # threshold value
        255,  # maxvalue
        cv2.THRESH_BINARY  # type
    )

    if channel == 'hue':
        # only works for filtering red color because the range for the hue is split
        channels['hue'] = cv2.bitwise_not(channels['hue'])


# get location of the laser given a frame
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

    # track
    mask = channels['laser']
    center = None
    countours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(countours) > 0:
        # print "found"
        c = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            center = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        else:
            center = int(x), int(y)
    # print center
    return center


# transform coordinates into directions for each engine and execute them
def pointsToDirections(laserArray, startLocation, startOrientation):
    directions = []  # tuples of format (t1, t2) where t1 = time for engine 1(left) to run, t2 = time for engine 2(right) to run.

    location = startLocation
    orientation = startOrientation
    for nextPoint in laserArray:
        if not (location == nextPoint):  # checks for zero vectors
            vector1 = orientation
        else: continue
        vector2 = (nextPoint[0] - location[0], nextPoint[1] - location[1])
        # get angle to turn, if any
        curAngleTime = angleTime(vector1, vector2)[0]
        if curAngleTime != (0, 0):
            executeDirections(curAngleTime)
            directions.append(curAngleTime)
            # get distance to move forward, if any
        curDistanceTime = math.hypot(vector2[0], vector2[1]) / SPEED
        if curDistanceTime != 0:
            executeDirections((curDistanceTime, curDistanceTime))
            directions.append((curDistanceTime,
                               curDistanceTime))  # investigate whether two motors at once gives different speed - must relate to rpm/SPEED issue
        orientation = vector2
        location = nextPoint  # changed location

    return directions  # execute directions given by function pointsToDirections.


def executeDirections(direction):
    print "exec"
    l = []
    if direction[0] != 0:
        l.append(Side1Enable)
        #l.append(p1)
    if direction[1] != 0:
        l.append(Side2Enable)
        #l.append(p2)

    GPIO.output(Side1Pin1, 1)
    GPIO.output(Side1Pin2, 0)
    GPIO.output(Side2Pin1, 0)
    GPIO.output(Side2Pin2, 1)

    #for pin in l: pin.start(100)

    GPIO.output(l, 1)
    executeTime = direction[0] if direction[0] != 0 else direction[1] 
    startTime = datetime.now()
    #while getDifference(startTime, datetime.now()) > direction[0]: continue
    time.sleep(executeTime)
    GPIO.output(l, 0)
    #for pin in l: pin.stop()


# simplify curve so that all deviations smaller than epsilon are omitted, should help with jitterings in pixels.
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

        resultList = recResults1[:-1] + recResults2  # to avoid having index twice.
    else:
        resultList = [pointList[0], pointList[end]]

    return resultList


# get perpendicular distance between a point and a line
def perpendicularDistance(point, line):
    # a = h*b/2 then h = 2a/b
    (l1, l2) = line
    a2 = math.fabs((l2[1] - l1[1]) * point[0] - (l2[0] - l1[0]) * point[1] + l2[0] * l1[1] - l2[1] * l1[0])
    b = math.hypot(l2[0] - l1[0], l2[1] - l1[1])
    return a2 / b


# transform angles into times for specific engine to run
def angleTime(v1, v2):
    revTime = 1.702 #time to revolve around itself
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    bottom = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
    angle = math.acos(dot / bottom)
    execTime = angle/(2 * math.pi) * revTime
    answer = (None, None)
    if (v1[0] * v2[1] - v1[1] * v2[0]) > 0:  # 2d-cross product tells you whether vector is to the right or left. negative = right positive = left
        answer = (0, execTime)
    else:
        answer = (execTime, 0)
        angle = -angle
    return answer, angle


def getAverage(pixel):
    average = 0
    for num in pixel:
        average += num
    return average / len(pixel[:3])
