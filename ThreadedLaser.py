import threading
from LaserFunctions import *

threads = []
t1 = threading.Thread(target=extractLocation)
threads.append(t1)
t2 = threading.Thread(target=pointsToDirection)
threads.append(t2)

for thread in threads:
    thread.start()
#need to figure out how to pass variables between two threads
