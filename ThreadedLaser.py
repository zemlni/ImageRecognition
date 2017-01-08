from multiprocessing import Process, Queue, Lock
from LaserFunctions import *
import time
from collections import deque

#robot is always at the middle bottom of the screen
start = (640/2, 0)
orientation = (0, 1)
time.sleep(5)
queue = Queue()
lock = Lock()
#queue = deque()
#reference frame updated on each frame
locationProcess = Process(target=extractLocation, args=(queue, lock))
execProcess = Process(target=move, args=(queue, lock))
processes = [locationProcess, execProcess]
try:
    for process in processes:
        process.start()
except KeyboardInterrupt:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([11, 12, 13, 15, 16, 18], GPIO.OUT)
    GPIO.output([11, 12, 13, 15, 16, 18], 0)
    GPIO.cleanup()
