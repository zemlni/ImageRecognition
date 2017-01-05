from multiprocessing import Process, Queue
from LaserFunctions import *

#robot is always at the middle bottom of the screen
start = (640/2, 0)
orientation = (0, 1)

queue = Queue()
#reference frame updated on each frame
locationProcess = Process(target=extractLocation, args=(queue,))
execProcess = Process(target=pointsToDirections, args=(queue, ))
processes = [locationProcess, execProcess]
for process in processes:
    process.start()
